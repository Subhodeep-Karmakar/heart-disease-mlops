"""
Unit tests for serialized model behavior and inference consistency.
"""

import os
import joblib
import pytest
import pandas as pd


@pytest.fixture
def loaded_pipeline():
    model_path = "models/best_model.pkl"
    # If model hasn't been trained yet, train it automatically
    if not os.path.exists(model_path):
        from src.train import train_and_track_models
        train_and_track_models()
        
    assert os.path.exists(model_path), f"Model artifact not found at {model_path}"
    pipeline = joblib.load(model_path)
    return pipeline


@pytest.fixture
def sample_patient():
    """A realistic sample patient record matching the UCI feature schema."""
    return pd.DataFrame([{
        "age": 58,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 211,
        "fbs": 1,
        "restecg": 2,
        "thalach": 165,
        "exang": 0,
        "oldpeak": 1.8,
        "slope": 2,
        "ca": 0.0,
        "thal": 3.0
    }])


def test_model_prediction_format(loaded_pipeline, sample_patient):
    """Test that the model predicts a valid binary class (0 or 1)."""
    prediction = loaded_pipeline.predict(sample_patient)
    assert len(prediction) == 1
    assert int(prediction[0]) in [0, 1]


def test_model_prediction_probabilities(loaded_pipeline, sample_patient):
    """Test that prediction probabilities are within [0, 1] and sum to 1."""
    probabilities = loaded_pipeline.predict_proba(sample_patient)[0]
    assert len(probabilities) == 2
    assert 0.0 <= probabilities[0] <= 1.0
    assert 0.0 <= probabilities[1] <= 1.0
    assert pytest.approx(probabilities.sum(), 0.001) == 1.0