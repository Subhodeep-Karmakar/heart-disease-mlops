"""
Model Training and MLflow Experiment Tracking Script.
Trains Logistic Regression and Random Forest models with 5-fold cross-validation,
evaluates metrics, logs runs to MLflow, and saves the best model.
"""

import sys
import os

# Ensure the root project directory is always in Python's import search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)

from src.preprocess import get_preprocessor, load_and_split_data


def train_and_track_models(data_path: str = "data/raw/heart.csv"):
    # 1. Load data
    X, y = load_and_split_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 2. Configure MLflow Experiment
    mlflow.set_experiment("Heart_Disease_Classification")
    
    # 3. Define candidate models to compare
    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "params": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"}
        },
        "Random_Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "params": {"n_estimators": 100, "max_depth": 5, "criterion": "gini"}
        }
    }
    
    best_roc_auc = 0.0
    best_pipeline = None
    best_model_name = ""
    os.makedirs("models", exist_ok=True)
    os.makedirs("docs/images", exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            print(f"\n{'='*20} Training {name} {'='*20}")
            
            # Create full pipeline: Preprocessor + Classifier
            preprocessor = get_preprocessor()
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", config["model"])
            ])
            
            # Log hyperparameters
            mlflow.log_params(config["params"])
            mlflow.log_param("cv_folds", 5)
            
            # 5-Fold Cross Validation on Training Set
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
            
            cv_accuracy = float(np.mean(cv_results["test_accuracy"]))
            cv_precision = float(np.mean(cv_results["test_precision"]))
            cv_recall = float(np.mean(cv_results["test_recall"]))
            cv_f1 = float(np.mean(cv_results["test_f1"]))
            cv_roc_auc = float(np.mean(cv_results["test_roc_auc"]))
            
            print(f"5-Fold CV Accuracy:  {cv_accuracy:.4f}")
            print(f"5-Fold CV Precision: {cv_precision:.4f}")
            print(f"5-Fold CV Recall:    {cv_recall:.4f}")
            print(f"5-Fold CV F1-Score:  {cv_f1:.4f}")
            print(f"5-Fold CV ROC-AUC:   {cv_roc_auc:.4f}")
            
            # Log CV Metrics
            mlflow.log_metric("cv_accuracy", cv_accuracy)
            mlflow.log_metric("cv_precision", cv_precision)
            mlflow.log_metric("cv_recall", cv_recall)
            mlflow.log_metric("cv_f1", cv_f1)
            mlflow.log_metric("cv_roc_auc", cv_roc_auc)
            
            # Fit on full training set and evaluate on held-out test set
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            test_accuracy = float(accuracy_score(y_test, y_pred))
            test_precision = float(precision_score(y_test, y_pred))
            test_recall = float(recall_score(y_test, y_pred))
            test_f1 = float(f1_score(y_test, y_pred))
            test_roc_auc = float(roc_auc_score(y_test, y_proba))
            
            print(f"Test Accuracy:       {test_accuracy:.4f}")
            print(f"Test ROC-AUC:        {test_roc_auc:.4f}")
            
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_roc_auc", test_roc_auc)
            
            # Generate and Log Confusion Matrix Artifact
            cm_fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
            plt.title(f"Confusion Matrix: {name}")
            cm_path = f"docs/images/cm_{name.lower()}.png"
            plt.savefig(cm_path, dpi=300, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(cm_path)
            
            # Generate and Log ROC Curve Artifact
            roc_fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
            plt.title(f"ROC Curve: {name}")
            roc_path = f"docs/images/roc_{name.lower()}.png"
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(roc_path)
            
            # Log Model Artifact to MLflow using cloudpickle
            mlflow.sklearn.log_model(
                pipeline, 
                name="model", 
                serialization_format="cloudpickle"
            )
            
            # Track best model based on ROC-AUC
            if test_roc_auc > best_roc_auc:
                best_roc_auc = test_roc_auc
                best_pipeline = pipeline
                best_model_name = name

    # 4. Save best pipeline as pickle for production API (Task 4)
    save_path = "models/best_model.pkl"
    joblib.dump(best_pipeline, save_path)
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name} with Test ROC-AUC = {best_roc_auc:.4f}")
    print(f"Final reproducible pipeline saved to: {save_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    train_and_track_models()