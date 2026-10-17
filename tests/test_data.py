"""
Unit tests for data acquisition and preprocessing pipeline.
"""

import os
import pytest
import pandas as pd
from src.preprocess import load_and_split_data, get_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture
def raw_data_path():
    path = "data/raw/heart.csv"
    assert os.path.exists(path), f"Raw dataset not found at {path}"
    return path


def test_data_loading_and_target_binarization(raw_data_path):
    """Test that X and y load correctly and target values are strictly 0 or 1."""
    X, y = load_and_split_data(raw_data_path)
    
    # Check dimensions
    assert len(X) == len(y), "Features and targets must have identical row counts"
    assert X.shape[1] == 13, f"Expected 13 feature columns, found {X.shape[1]}"
    
    # Check binary target
    unique_targets = set(y.unique())
    assert unique_targets.issubset({0, 1}), f"Target must be binary {0, 1}, got {unique_targets}"


def test_preprocessor_output_shape(raw_data_path):
    """Test that the ColumnTransformer runs and outputs clean numeric arrays without NaNs."""
    X, y = load_and_split_data(raw_data_path)
    preprocessor = get_preprocessor()
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Verify no NaN values remain after imputation
    assert not pd.isna(X_transformed).any(), "Transformed data must not contain any NaNs"
    assert X_transformed.shape[0] == X.shape[0], "Row count must remain invariant"