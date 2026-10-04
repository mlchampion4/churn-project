import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.data.load import load_data


NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_charges", "num_services",
]

BINARY_FEATURES = [
    "SeniorCitizen", "is_new_client", "has_support", "is_charge_anomaly",
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["avg_charges"] = df["TotalCharges"] / (df["tenure"] + 1)

    services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = (df[services] == "Yes").sum(axis=1)

    df["is_new_client"] = (df["tenure"] < 6).astype(int)

    df["has_support"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-1y", "1-2y", "2-4y", "4-6y"],
    ).astype(str)

    expected = df["tenure"] * df["MonthlyCharges"]
    diff_pct = (df["TotalCharges"] - expected).abs() / expected.replace(0, np.nan)
    df["is_charge_anomaly"] = (diff_pct > 0.5).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


if __name__ == "__main__":
    df = add_features(load_data())
    print(f"Shape after FE: {df.shape}")
    print(f"New columns: {[c for c in df.columns if c not in load_data().columns]}")

    preprocessor = build_preprocessor()
    X = df.drop(columns=["Churn"])
    X_transformed = preprocessor.fit_transform(X)
    print(f"After preprocessor: {X_transformed.shape}")

    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    print(f"Saved: {models_dir / 'preprocessor.pkl'}")