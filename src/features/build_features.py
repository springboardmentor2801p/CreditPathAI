import pandas as pd
import numpy as np

from src.data.db import load_from_db
from src.data.clean_data import clean_data


def build_features(df):

    # ================================
    # 🔹 Feature Engineering
    # ================================

    df["repayment_capacity"] = df["income"] / (df["loan_amount"] + 1)

    df["loan_burden_ratio"] = df["loan_amount"] / (df["income"] + 1)

    df["income_to_loan_ratio"] = df["income"] / (df["loan_amount"] + 1)
    df["credit_score_scaled"] = df["credit_score"] / 850

    df["financial_stress_index"] = df["ltv"] * df["dtir1"]
    df["financial_strength"] = (df["income"] * df["credit_score"]) / (df["loan_amount"] + 1)

    df["loan_amount_log"] = np.log1p(df["loan_amount"])
    df["income_log"] = np.log1p(df["income"])

    # 🔥 NEW FEATURE (ADD THIS LINE)
    df["risk_combined"] = df["ltv"] * df["dtir1"] / (df["credit_score"] + 1)

    print("\n✅ Features created")

    return df


if __name__ == "__main__":

    # 1. Load from DB
    df = load_from_db()

    # 2. Clean data
    df = clean_data(df)

    # 3. Feature engineering
    df = build_features(df)

    print("\nFinal Data Shape:", df.shape)
    print(df.head())