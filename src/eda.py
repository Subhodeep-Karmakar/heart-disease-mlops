"""
Exploratory Data Analysis (EDA) Script.
Performs data inspection, target binarization, summary statistics,
and generates publication-quality figures for the project report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(input_path: str = "data/raw/heart.csv", output_dir: str = "docs/images") -> None:
    # 1. Load Data
    df = pd.read_csv(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("\nData Types and Non-Null Counts:")
    print(df.info())
    
    print("\nMissing Values per Column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
    
    # 2. Binarize Target (0 = Healthy, 1+ = Disease Present)
    # The UCI target column is named 'num'
    target_col = "num" if "num" in df.columns else df.columns[-1]
    df["target"] = (df[target_col] > 0).astype(int)
    
    print("\nTarget Distribution:")
    print(df["target"].value_counts(normalize=True).rename({0: "Healthy (0)", 1: "Disease (1)"}))
    
    # Styling for professional plots
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # -------------------------------------------------------------
    # Plot 1: Target Class Balance
    # -------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="target", data=df, hue="target", palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Target Distribution: Absence vs Presence of Heart Disease", fontsize=13, weight="bold")
    plt.xlabel("Diagnosis (0 = Absence, 1 = Presence)")
    plt.ylabel("Patient Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", weight="bold", fontsize=12)
    plt.savefig(f"{output_dir}/01_class_balance.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/01_class_balance.png")

    # -------------------------------------------------------------
    # Plot 2: Numerical Distributions
    # -------------------------------------------------------------
    num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=axes[i], palette=["#2ecc71", "#e74c3c"])
        axes[i].set_title(f"Distribution of {col}", weight="bold")
    
    # Remove unused 6th subplot
    fig.delaxes(axes[5])
    plt.savefig(f"{output_dir}/02_numerical_distributions.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/02_numerical_distributions.png")

    # -------------------------------------------------------------
    # Plot 3: Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    corr = df[num_cols + ["target"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap (Continuous Features & Target)", fontsize=13, weight="bold")
    plt.savefig(f"{output_dir}/03_correlation_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/03_correlation_heatmap.png")

    # -------------------------------------------------------------
    # Plot 4: Key Categorical Features vs Target (Chest Pain & Sex)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.countplot(data=df, x="cp", hue="target", ax=ax1, palette=["#2ecc71", "#e74c3c"])
    ax1.set_title("Heart Disease Risk by Chest Pain Type (cp)", weight="bold")
    ax1.set_xlabel("Chest Pain Type (1-4)")
    ax1.set_ylabel("Count")

    sns.countplot(data=df, x="sex", hue="target", ax=ax2, palette=["#2ecc71", "#e74c3c"])
    ax2.set_title("Heart Disease Risk by Sex (0 = Female, 1 = Male)", weight="bold")
    ax2.set_xlabel("Sex")
    ax2.set_ylabel("Count")

    plt.savefig(f"{output_dir}/04_categorical_risk_factors.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir}/04_categorical_risk_factors.png")
    
    print("\nEDA completed! All 4 figures generated successfully in", output_dir)


if __name__ == "__main__":
    perform_eda()