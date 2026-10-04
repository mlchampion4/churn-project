# Telco Churn Prediction

## Бизнес-задача
Снизить отток клиентов на 3 п.п. через ML-скоринг + uplift.

## Метрики
- ML: ROC-AUC 0.87, PR-AUC 0.69
- Бизнес: экономия ~$X/мес при удержании 20% топ-риска

## Архитектура
[схема: CSV → features → LGBM → FastAPI → бизнес]

## Как запустить
make install && make train && make api

## Результаты A/B
- Control churn: 26.5%
- Treatment churn: 23.5% (p < 0.01)