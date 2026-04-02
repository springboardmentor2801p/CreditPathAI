import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.db import load_from_db
from src.data.clean_data import clean_data


def run_eda():

    df = load_from_db()
    df = clean_data(df)

    # ================================
    # 🔹 TARGET DISTRIBUTION
    # ================================
    plt.figure()
    sns.countplot(x="status", data=df)
    plt.title("Target Distribution (Default vs Non-Default)")
    plt.show()

    print("\nTarget Distribution:")
    print(df["status"].value_counts(normalize=True))

    # ================================
    # 🔹 NUMERICAL FEATURES
    # ================================
    num_cols = ["loan_amount", "income", "credit_score", "ltv", "dtir1"]

    for col in num_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()

    # ================================
    # 🔹 RELATION WITH TARGET
    # ================================
    for col in num_cols:
        plt.figure()
        sns.boxplot(x="status", y=col, data=df)
        plt.title(f"{col} vs Default")
        plt.show()

    # ================================
    # 🔹 CORRELATION HEATMAP
    # ================================
    plt.figure()
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # ================================
    # 🔹 CATEGORICAL ANALYSIS
    # ================================
    cat_cols = ["loan_type", "region", "occupancy_type"]

    for col in cat_cols:
        plt.figure()
        sns.countplot(x=col, hue="status", data=df)
        plt.title(f"{col} vs Default")
        plt.xticks(rotation=45)
        plt.show()


if __name__ == "__main__":
    run_eda()