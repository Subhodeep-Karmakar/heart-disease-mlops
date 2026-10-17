"""
FastAPI Model Serving Application with Prometheus Monitoring & Request Logging.
Provides /health, /predict, and /metrics endpoints.
"""

import os
import sys
import time
import logging
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Structured Logging Setup (Task 8)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production MLOps inference API with Prometheus monitoring.",
    version="1.0.0"
)

# 2. Prometheus Metrics Definitions (Task 8)
REQUEST_COUNT = Counter(
    "api_requests_total", 
    "Total count of requests to the API", 
    ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", 
    "Latency of API requests in seconds", 
    ["endpoint"]
)
PREDICTIONS_BY_CLASS = Counter(
    "model_predictions_total", 
    "Total predictions classified by outcome", 
    ["prediction_class", "risk_level"]
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
pipeline = None


@app.on_event("startup")
def load_model():
    """Loads the serialized model pipeline when API starts up."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model artifact not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file missing at {MODEL_PATH}. Run training first.")
    
    pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Model successfully loaded from {MODEL_PATH}")


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, example=58)
    sex: int = Field(..., ge=0, le=1, example=1)
    cp: int = Field(..., ge=1, le=4, example=4)
    trestbps: int = Field(..., ge=50, le=250, example=140)
    chol: int = Field(..., ge=100, le=600, example=211)
    fbs: int = Field(..., ge=0, le=1, example=1)
    restecg: int = Field(..., ge=0, le=2, example=2)
    thalach: int = Field(..., ge=50, le=250, example=165)
    exang: int = Field(..., ge=0, le=1, example=0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, example=1.8)
    slope: int = Field(..., ge=1, le=3, example=2)
    ca: float = Field(..., ge=0.0, le=3.0, example=0.0)
    thal: float = Field(..., ge=3.0, le=7.0, example=3.0)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    risk_level: str
    processing_time_ms: float


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "model_path": MODEL_PATH
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientData) -> PredictionResponse:
    if pipeline is None:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="503").inc()
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    start_time = time.time()
    input_df = pd.DataFrame([patient.dict()])
    
    try:
        prediction_class = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction_class])
        
        duration = time.time() - start_time
        proc_time = round(duration * 1000, 2)
        
        label = "Presence of Heart Disease" if prediction_class == 1 else "Absence of Heart Disease"
        risk = "HIGH RISK" if prediction_class == 1 else "LOW RISK"
        
        # 1. Structured log entry
        logger.info(
            f"Prediction: {prediction_class} ({label}) | "
            f"Confidence: {confidence:.4f} | Latency: {proc_time}ms"
        )
        
        # 2. Record Prometheus metrics
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(duration)
        PREDICTIONS_BY_CLASS.labels(prediction_class=str(prediction_class), risk_level=risk).inc()
        
        return PredictionResponse(
            prediction=prediction_class,
            prediction_label=label,
            confidence=round(confidence, 4),
            risk_level=risk,
            processing_time_ms=proc_time
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")