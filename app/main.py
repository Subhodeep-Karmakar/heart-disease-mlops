"""
FastAPI Model Serving Application.
Provides /health and /predict endpoints for real-time Heart Disease risk prediction.
Includes Pydantic schema validation, confidence calculation, and request logging.
"""

import os
import sys
import time
import logging
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure structured logging for API monitoring (Task 8)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production MLOps inference API using trained Scikit-Learn Pipeline.",
    version="1.0.0"
)

# Global model holder
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
pipeline = None


@app.on_event("startup")
def load_model():
    """Loads the serialized MLflow/pickle pipeline when API starts up."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model artifact not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file missing at {MODEL_PATH}. Run training first.")
    
    pipeline = joblib.load(MODEL_PATH)
    logger.info(f"Model successfully loaded from {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Pydantic Schemas for Input Validation
# ---------------------------------------------------------------------------
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years (e.g. 58)", example=58)
    sex: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male", example=1)
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)", example=4)
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)", example=140)
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)", example=211)
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)", example=1)
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)", example=2)
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved", example=165)
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = Yes, 0 = No)", example=0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise", example=1.8)
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)", example=2)
    ca: float = Field(..., ge=0.0, le=3.0, description="Number of major vessels (0-3)", example=0.0)
    thal: float = Field(..., ge=3.0, le=7.0, description="Defect type (3 = normal, 6 = fixed, 7 = reversible)", example=3.0)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Absence of Disease, 1 = Presence of Disease")
    prediction_label: str
    confidence: float
    risk_level: str
    processing_time_ms: float


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Healthcheck endpoint for Kubernetes Liveness and Readiness probes."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientData) -> PredictionResponse:
    """Accepts patient data as JSON and returns heart disease prediction & confidence."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded.")
    
    start_time = time.time()
    
    # Convert validated Pydantic model to Pandas DataFrame
    input_df = pd.DataFrame([patient.dict()])
    
    try:
        # Run inference through full pipeline (imputer + scaler + onehot + classifier)
        prediction_class = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction_class])
        
        proc_time = round((time.time() - start_time) * 1000, 2)
        
        label = "Presence of Heart Disease" if prediction_class == 1 else "Absence of Heart Disease"
        risk = "HIGH RISK" if prediction_class == 1 else "LOW RISK"
        
        # Log inference event for monitoring (Task 8)
        logger.info(
            f"Prediction: {prediction_class} ({label}) | "
            f"Confidence: {confidence:.4f} | Latency: {proc_time}ms"
        )
        
        return PredictionResponse(
            prediction=prediction_class,
            prediction_label=label,
            confidence=round(confidence, 4),
            risk_level=risk,
            processing_time_ms=proc_time
        )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")