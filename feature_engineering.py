import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def process_data(df):

    # Separate columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'str']).columns

    # Fill numerical missing values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encoding
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Feature extraction
    df["loan_income_ratio"] = df["loan_amount"] / df["income"]
    df["interest_burden"] = df["loan_amount"] * df["rate_of_interest"]

    # Save cleaned data
    df.to_csv("data/processed/cleaned.csv", index=False)

    return df
from data_ingestion import load_data

df = load_data()
process_data(df)
