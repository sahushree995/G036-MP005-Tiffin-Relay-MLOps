from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
FEATURES = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
    "transport",
    "route_zone",
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: float
    tiffins: Annotated[int, Field(le=100)]
    handovers: int
    wait_at_station_min: float
    rider_age: int
    temp_c: Annotated[float, Field(ge=5)]
    transport: str
    route_zone: str


class PredictionResponse(BaseModel):
    prediction_minutes: float


app = FastAPI(title="Tiffin Relay Prediction API", version="1.0.0")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = pd.DataFrame([request.model_dump()])[FEATURES]
    prediction = round(float(model.predict(features)[0]), 1)
    return PredictionResponse(prediction_minutes=prediction)
