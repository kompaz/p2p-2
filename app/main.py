import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import CustomerData


MODEL_PATH = "./artifacts/model_pipeline.pkl"

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running."}


@app.post("/predict")
def predict(customer: CustomerData):
    input_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "prediction": int(prediction),
        "prediction_label": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": float(probability)
    }