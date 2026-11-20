import os
import subprocess
import pandas as pd
import mlflow
import mlflow.sklearn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "VehiclePriceEstimator")
MODEL_STAGE_OR_VERSION = os.getenv("MODEL_VERSION", "1")

ml_models = {}

def load_registered_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE_OR_VERSION}"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback: run Phase 1 pipeline if DB is fresh (e.g., inside CI runner)
        subprocess.run(["python", "src/ingest.py"], check=True)
        subprocess.run(["python", "src/preprocess.py"], check=True)
        subprocess.run(["python", "src/train_evaluate.py"], check=True)
        return mlflow.sklearn.load_model(model_uri)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["estimator"] = load_registered_model()
    yield
    ml_models.clear()

app = FastAPI(
    title="Vehicle Price Estimation API",
    description="Phase 2 Production Inference Service backed by MLflow Model Registry",
    version="2.0.0",
    lifespan=lifespan,
)

class VehicleFeatures(BaseModel):
    year: int = Field(..., ge=1990, le=2026, example=2019)
    present_price: float = Field(..., gt=0, example=8.5)
    kms_driven: int = Field(..., ge=0, example=32000)
    fuel_type: str = Field(..., example="Petrol")
    seller_type: str = Field(..., example="Dealer")
    transmission: str = Field(..., example="Manual")
    owner: int = Field(..., ge=0, le=3, example=0)

class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    model_name: str
    model_version: str

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "estimator" in ml_models,
        "model_name": MODEL_NAME,
        "model_version": MODEL_STAGE_OR_VERSION,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_price(vehicle: VehicleFeatures):
    if "estimator" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Feature engineering matching Phase 1 preprocess.py
    vehicle_age = 2026 - vehicle.year
    km_per_year = vehicle.kms_driven / max(vehicle_age, 1)

    input_df = pd.DataFrame([{
        "Present_Price": vehicle.present_price,
        "Kms_Driven": vehicle.kms_driven,
        "Owner": vehicle.owner,
        "vehicle_age": vehicle_age,
        "km_per_year": km_per_year,
        "Fuel_Type_Diesel": 1 if vehicle.fuel_type.lower() == "diesel" else 0,
        "Fuel_Type_Petrol": 1 if vehicle.fuel_type.lower() == "petrol" else 0,
        "Seller_Type_Individual": 1 if vehicle.seller_type.lower() == "individual" else 0,
        "Transmission_Manual": 1 if vehicle.transmission.lower() == "manual" else 0,
    }])

    # Align columns with model's expected training features
    model = ml_models["estimator"]
    if hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = float(model.predict(input_df)[0])
    return PredictionResponse(
        predicted_price_lakhs=round(prediction, 2),
        model_name=MODEL_NAME,
        model_version=MODEL_STAGE_OR_VERSION,
    )