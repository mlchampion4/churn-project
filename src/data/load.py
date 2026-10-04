from pathlib import Path
import pandas as pd


DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["customerID"])
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.3f}")
    print(f"NaN в TotalCharges: {df['TotalCharges'].isna().sum()}")
    print(df.head())