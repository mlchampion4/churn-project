import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_response(client):
    payload = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5, "TotalCharges": 427.5,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk"] in ("low", "medium", "high")
    assert len(body["top_factors"]) == 3


def test_predict_batch(client):
    payload = {
        "clients": [
            {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5, "TotalCharges": 427.5,
            },
            {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 60, "PhoneService": "Yes",
                "MultipleLines": "Yes", "InternetService": "DSL",
                "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
                "DeviceProtection": "Yes", "TechSupport": "Yes",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Two year", "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
            },
        ]
    }
    r = client.post("/predict_batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2