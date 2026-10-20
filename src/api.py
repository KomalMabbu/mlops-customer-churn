
import mlflow
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_pipeline"
MODEL_RUN_ID = "18d01f2992e044ada0f7374478846ebc"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0.0",
)

# Load the trained preprocessing + prediction pipeline.
model = mlflow.sklearn.load_model(str(MODEL_PATH))

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    try:
        # The model expects the original 19 features, excluding customerID.
        customer_data = pd.DataFrame([customer.model_dump()])

        prediction = int(model.predict(customer_data)[0])
        probability = float(model.predict_proba(customer_data)[0][1])

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(probability, 4),
            "model_run_id": MODEL_RUN_ID,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc