"""
Data Acquisition Script for Heart Disease UCI Dataset.
Fetches the dataset from the UCI Machine Learning Repository
and saves the raw data to data/raw/heart.csv.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_heart_disease_data(output_path: str = "data/raw/heart.csv") -> pd.DataFrame:
    """
    Downloads Heart Disease dataset from UCI repository (ID: 45)
    and saves it to the specified CSV path.
    """
    print("Fetching Heart Disease dataset from UCI repository (ID: 45)...")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fetch dataset using official ucimlrepo
    heart_disease = fetch_ucirepo(id=45)
    
    # Extract features (X) and target (y)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Combine features and target into a single DataFrame
    df = pd.concat([X, y], axis=1)
    
    # Save raw CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully downloaded and saved to: {output_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    download_heart_disease_data()