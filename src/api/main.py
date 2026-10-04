from pathlib import Path
from contextlib import asynccontextmanager
import json

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build import add_features


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

_state: dict = {}


def _dummy_client() -> dict:
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 50.0, "TotalCharges": 50.0,
    }


def _prepare(client: dict) -> pd.DataFrame:
    df = pd.DataFrame([client])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return add_features(df)


def _prepare_batch(clients: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(clients)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return add_features(df)


def _recommend(risk: str) -> str:
    if risk == "high":
        return "Предложить скидку 15% на годовую подписку + бесплатный tech support на 3 месяца"
    if risk == "medium":
        return "Отправить напоминание о сервисе, проверить вовлечённость"
    return "Не беспокоить — клиент лоялен"


def _risk_level(proba: float, threshold: float) -> str:
    if proba >= threshold:
        return "high"
    if proba >= threshold * 0.5:
        return "medium"
    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["model"] = joblib.load(MODEL_PATH)
    _state["preprocessor"] = _state["model"].named_steps["prep"]
    _state["clf"] = _state["model"].named_steps["clf"]
    _state["threshold"] = 0.32

    dummy_X = _prepare(_dummy_client())
    dummy_transformed = _state["preprocessor"].transform(dummy_X)

    _state["explainer"] = shap.LinearExplainer(_state["clf"], dummy_transformed)
    _state["feature_names"] = _state["preprocessor"].get_feature_names_out()

    yield
    _state.clear()


app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="Прогноз оттока клиентов телеком-компании + SHAP-объяснения",
    lifespan=lifespan,
)


class Client(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    risk: str
    threshold: float
    recommendation: str
    top_factors: list[dict]


class BatchRequest(BaseModel):
    clients: list[Client]


def _explain(X: pd.DataFrame, top_k: int = 3) -> list[dict]:
    X_transformed = _state["preprocessor"].transform(X)
    shap_values = _state["explainer"].shap_values(X_transformed)
    feature_names = _state["feature_names"]

    sv = shap_values[0]
    top_idx = np.argsort(np.abs(sv))[::-1][:top_k]

    factors = []
    for idx in top_idx:
        factors.append({
            "feature": str(feature_names[idx]),
            "shap_value": round(float(sv[idx]), 4),
            "direction": "to_churn" if sv[idx] > 0 else "from_churn",
        })
    return factors


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": type(_state["clf"]).__name__,
        "threshold": _state["threshold"],
    }


@app.get("/metadata")
def metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="metadata.json не найден")
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
def predict(client: Client):
    X = _prepare(client.model_dump())
    proba = float(_state["model"].predict_proba(X)[:, 1][0])
    risk = _risk_level(proba, _state["threshold"])
    factors = _explain(X, top_k=3)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        risk=risk,
        threshold=_state["threshold"],
        recommendation=_recommend(risk),
        top_factors=factors,
    )


@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    X = _prepare_batch([c.model_dump() for c in request.clients])
    probas = _state["model"].predict_proba(X)[:, 1]
    threshold = _state["threshold"]

    results = []
    for proba in probas:
        risk = _risk_level(float(proba), threshold)
        results.append({
            "churn_probability": round(float(proba), 4),
            "risk": risk,
            "recommendation": _recommend(risk),
        })
    return {"predictions": results, "count": len(results)}