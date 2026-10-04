import pandas as pd
import pytest
from src.features.build import add_features, build_preprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gender": ["Female"], "SeniorCitizen": [0], "Partner": ["Yes"],
        "Dependents": ["No"], "tenure": [10], "PhoneService": ["Yes"],
        "MultipleLines": ["No"], "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"], "OnlineBackup": ["No"],
        "DeviceProtection": ["Yes"], "TechSupport": ["No"],
        "StreamingTV": ["Yes"], "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"], "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [50.0], "TotalCharges": [500.0],
    })


def test_add_features_creates_columns(sample_df):
    out = add_features(sample_df)
    expected = {"avg_charges", "num_services", "is_new_client",
                "has_support", "tenure_group", "is_charge_anomaly"}
    assert expected.issubset(out.columns)


def test_num_services_counts_correctly(sample_df):
    out = add_features(sample_df)
    assert out["num_services"].iloc[0] == 3


def test_has_support(sample_df):
    out = add_features(sample_df)
    assert out["has_support"].iloc[0] == 1


def test_preprocessor_shape():
    from src.data.load import load_data
    df = add_features(load_data()).drop(columns=["Churn"])
    prep = build_preprocessor()
    X = prep.fit_transform(df.head(100))
    assert X.shape[0] == 100
    assert X.shape[1] > 30