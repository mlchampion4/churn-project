from pathlib import Path
import json
import joblib
import pandas as pd

from src.features.build import add_features


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}. Запусти python -m src.models.train"
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {"optimal_threshold": 0.5}
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _prepare_features(client: dict) -> pd.DataFrame:
    df = pd.DataFrame([client])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return add_features(df)


def predict_one(client: dict) -> dict:
    model = load_model()
    metadata = load_metadata()
    threshold = metadata.get("optimal_threshold", 0.5)

    X = _prepare_features(client)
    proba = float(model.predict_proba(X)[:, 1][0])

    if proba >= threshold:
        risk = "high"
    elif proba >= threshold * 0.5:
        risk = "medium"
    else:
        risk = "low"

    return {
        "churn_probability": round(proba, 4),
        "risk": risk,
        "threshold": threshold,
        "recommendation": _recommend(risk),
    }


def _recommend(risk: str) -> str:
    if risk == "high":
        return "Предложить скидку 15% на годовую подписку + tech support"
    if risk == "medium":
        return "Проверить вовлечённость, отправить напоминание о сервисе"
    return "Не беспокоить — клиент лоялен"


def predict_batch(clients: list[dict]) -> list[dict]:
    model = load_model()
    metadata = load_metadata()
    threshold = metadata.get("optimal_threshold", 0.5)

    df = pd.DataFrame(clients)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = add_features(df)

    probas = model.predict_proba(df)[:, 1]

    results = []
    for proba in probas:
        risk = (
            "high" if proba >= threshold
            else "medium" if proba >= threshold * 0.5
            else "low"
        )
        results.append({
            "churn_probability": round(float(proba), 4),
            "risk": risk,
            "threshold": threshold,
        })
    return results


if __name__ == "__main__":
    sample_client = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5,
    }

    print(json.dumps(predict_one(sample_client), indent=2, ensure_ascii=False))