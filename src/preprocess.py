"""
Preprocessing Pipeline Module.
Builds a reusable Scikit-Learn ColumnTransformer for numeric and categorical features.
Ensures zero data leakage and full reproducibility.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Feature definitions based on Cleveland Heart Disease dataset
NUMERICAL_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL: str = "num"


def get_preprocessor() -> ColumnTransformer:
    """
    Creates a ColumnTransformer that:
      1. Imputes missing numeric values with median, then applies StandardScaler.
      2. Imputes missing categorical values with most frequent, then applies OneHotEncoder.
    """
    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Bundle both into a single ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor


def load_and_split_data(csv_path: str = "data/raw/heart.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw CSV, creates binary target, and returns X and y.
    """
    df = pd.read_csv(csv_path)
    
    # Binarize target: 0 = healthy, >=1 = heart disease
    y = (df[TARGET_COL] > 0).astype(int)
    X = df.drop(columns=[TARGET_COL])
    
    return X, y