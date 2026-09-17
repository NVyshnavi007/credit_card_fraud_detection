from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import random

app = FastAPI(title="Credit Card Fraud Detection API")

model = joblib.load('fraud_model.joblib')
demo_data = pd.read_csv('demo_transactions.csv')

FEATURE_COLS = [c for c in demo_data.columns if c != 'actual_class']

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/random-transaction")
def random_transaction():
    row = demo_data.sample(1).iloc[0]
    transaction_id = int(row.name)
    return {
        "transaction_id": transaction_id,
        "actual_class": int(row['actual_class']),
        "features": row[FEATURE_COLS].to_dict()
    }

@app.get("/predict/{transaction_id}")
def predict(transaction_id: int):
    if transaction_id < 0 or transaction_id >= len(demo_data):
        raise HTTPException(status_code=404, detail="Transaction ID not found")

    row = demo_data.iloc[transaction_id]
    features = row[FEATURE_COLS].values.reshape(1, -1)

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "transaction_id": transaction_id,
        "predicted_class": prediction,
        "fraud_probability": round(probability, 4),
        "actual_class": int(row['actual_class']),
        "correct": prediction == int(row['actual_class'])
    }