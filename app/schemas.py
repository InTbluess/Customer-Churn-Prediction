from fastapi import FastAPI

from app.schemas import Customer

from src.pipeline import predict


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is Running!"
    }


@app.post("/predict")
def predict_customer(customer: Customer):

    result = predict(customer.model_dump())

    return result