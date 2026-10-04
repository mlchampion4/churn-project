# 🎯 Telco Churn Prediction — End-to-End ML System

Полный ML-пайплайн: EDA → feature engineering → сравнение моделей
с Optuna → SHAP → FastAPI → Docker → тесты.

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

Bootstrap 95% CI пересекаются → модели статистически не отличаются.
Выбрал LogReg: минимальный gap, простота, скорость, интерпретируемость.

## 🏗️ Архитектура
