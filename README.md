# Credit Card Fraud Detection API

An end-to-end machine learning project that detects fraudulent credit card transactions in a severely imbalanced dataset (0.17% fraud), deployed as a live REST API.

**Live Demo:** [https://fraud-detection-api-r37w.onrender.com/docs](https://fraud-detection-api-r37w.onrender.com/docs)
*(First request may take 20–30 seconds — free-tier hosting spins down when idle.)*

## Problem

Standard accuracy is meaningless on this dataset: predicting "not fraud" for every transaction scores 99.83% accuracy while catching zero fraud. This project focuses on building and evaluating models properly for extreme class imbalance, rather than chasing a misleading metric.

## Dataset

- Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions, 492 fraudulent (0.17%)
- Features `V1`–`V28` are PCA-transformed (anonymized); `Time` and `Amount` are raw

## Approach

1. **EDA** — confirmed severe class imbalance; compared `Amount` distributions between fraud and normal transactions (fraud has a lower median but higher mean, i.e. mostly small transactions with some larger outliers)
2. **Preprocessing** — scaled `Amount` and `Time` with `StandardScaler` to match the PCA-scaled `V1`–`V28` features
3. **Stratified train/test split** — preserved the 99.83% / 0.17% ratio in both sets
4. **Baseline model** — plain Logistic Regression: high accuracy (1.00) but weak fraud recall (0.64), demonstrating why accuracy is the wrong metric here
5. **Imbalance handling** — `class_weight='balanced'` improved recall (0.92) but collapsed precision (0.06) — a real precision/recall trade-off, not a bug
6. **Model comparison** — Random Forest and XGBoost handled the imbalance far better than Logistic Regression without that trade-off:

   | Model | Precision (fraud) | Recall (fraud) | AP score |
   |---|---|---|---|
   | Logistic Regression (balanced) | 0.06 | 0.92 | 0.719 |
   | Random Forest (balanced) | 0.96 | 0.76 | 0.865 |
   | XGBoost | 0.89 | 0.83 | 0.873 |

7. **Tuning** — Stratified 5-Fold CV + GridSearchCV on XGBoost (`n_estimators`, `max_depth`, `learning_rate`), optimizing for Average Precision rather than accuracy
8. **Final model** — XGBoost, tuned: **0.84 precision / 0.83 recall / 0.865 AP** on a held-out test set

## Why Average Precision (AP), not accuracy

AP summarizes precision and recall across every possible classification threshold, rather than one arbitrary cutoff (0.5). It's a more honest metric for ranking model quality on imbalanced data, where accuracy is misleading and a single precision/recall pair depends on threshold choice.

## API

Built with FastAPI, serving a small demo sample (298 transactions: all 98 fraud cases + 200 random normal ones) since the raw PCA features aren't human-readable.

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /random-transaction` | Returns a random transaction's features (label hidden) |
| `GET /predict/{transaction_id}` | Returns the model's fraud prediction, probability, and whether it matched the true label |

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · FastAPI · Joblib · Render (deployment)

## Limitations

- `V1`–`V28` are PCA-transformed upstream by the dataset provider — PCA is unsupervised and doesn't account for the fraud label when deciding which directions to keep, so it could in principle under-weight rare-but-decisive signals
- Trained on a single historical dataset; fraud patterns evolve over time, so a production system would need periodic retraining and drift monitoring
