# 🎯 Telco Churn Prediction — End-to-End ML System

Полный ML-пайплайн: EDA → feature engineering → сравнение моделей с Optuna → SHAP → FastAPI → Docker → тесты.

## 📊 Метрики финальной модели

**Модель:** LogisticRegression · **Порог:** 0.32 (по бизнес-метрике)

| Метрика | Значение |
|---|---|
| ROC-AUC (test) | **0.8476** |
| PR-AUC | 0.6614 |
| Gap (train-test) | **0.0055** |
| Recall (churn) | **0.92** |
| Precision (churn) | 0.44 |
| Expected profit | **$47,150/мес** |

## 🎯 Почему LogReg, а не CatBoost

| Модель | Test ROC-AUC | Gap |
|---|---|---|
| **LogReg** | **0.8476** | **0.0055** |
| CatBoost (Optuna) | 0.8470 | 0.0149 |
| LightGBM (Optuna) | 0.8443 | 0.0375 |

Bootstrap 95% CI пересекаются → модели статистически не отличаются. Выбрал LogReg: минимальный gap, простота, скорость, интерпретируемость.

## 🏗️ Архитектура

CSV → features → LogReg → FastAPI → SHAP → CRM

## 🚀 Быстрый старт

git clone https://github.com/USERNAME/churn-project.git
cd churn-project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.models.train
pytest tests/ -v
uvicorn src.api.main:app --reload

API: http://localhost:8000/docs

## 🐳 Docker

docker build -t churn-api -f docker/Dockerfile .
docker run -p 8000:8000 churn-api

## 🔌 Пример запроса

POST /predict — тело запроса:

{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
  "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
  "MultipleLines": "No", "InternetService": "Fiber optic",
  "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "No",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.5, "TotalCharges": 427.5
}

Ответ:

{
  "churn_probability": 0.8231,
  "risk": "high",
  "threshold": 0.32,
  "recommendation": "Предложить скидку 15% на годовую подписку + бесплатный tech support на 3 месяца",
  "top_factors": [
    {"feature": "num__avg_charges", "shap_value": 0.42, "direction": "to_churn"},
    {"feature": "cat__Contract_Month-to-month", "shap_value": 0.31, "direction": "to_churn"},
    {"feature": "num__tenure", "shap_value": -0.22, "direction": "from_churn"}
  ]
}

## 📁 Структура

churn-project/
├── src/
│   ├── data/load.py
│   ├── features/build.py
│   ├── models/{train,predict}.py
│   └── api/main.py
├── tests/
├── docker/
├── notebooks/
├── models/
├── requirements.txt
└── README.md

## 🛠️ Стек

pandas · scikit-learn · SHAP · FastAPI · Docker · pytest · Optuna

## 👤 Автор

Ваше Имя · LinkedIn: https://linkedin.com/in/USERNAME · GitHub: https://github.com/USERNAME