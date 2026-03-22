# ===============================
# DATA PREPROCESSING
# CreditPath AI
# ===============================

import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the loan dataset from CSV file"""
    df = pd.read_csv(filepath)
    print("Original Shape:", df.shape)
    return df

def clean_data(df):
    """Clean and prepare the dataset"""

    # Drop ID column if exists
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Remove leakage columns
    leak_cols = [
        "credit_type",
        "co-applicant_credit_type",
        "Interest_rate_spread",
        "Upfront_charges",
        "rate_of_interest"
    ]
    df = df.drop(columns=[col for col in leak_cols if col in df.columns])

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    print("After Removing Duplicates:", df.shape)

    # Handle missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df

def feature_engineering(df):
    """Create new features from existing ones"""

    def safe_divide(a, b):
        return a / (b + 1)

    # Core ratio features
    if "loan_amount" in df.columns and "income" in df.columns:
        df["loan_to_income_ratio"] = safe_divide(df["loan_amount"], df["income"])
        df["payment_to_income_ratio"] = safe_divide(df["loan_amount"], df["income"])

    # Flag features
    if "dtir1" in df.columns:
        df["high_dti_flag"] = (df["dtir1"] > 40).astype(int)

    if "Credit_Score" in df.columns:
        df["low_credit_flag"] = (df["Credit_Score"] < 650).astype(int)

    print("Features after engineering:", df.shape[1])
     # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    return df

def get_features_and_target(df):
    

    simple_features = [
    'Credit_Score', 'loan_amount', 'income', 'LTV', 'dtir1',
    'loan_to_income_ratio', 'payment_to_income_ratio',
    'high_dti_flag', 'low_credit_flag', 'property_value',
    'term', 'age', 'Neg_ammortization', 'occupancy_type',
    'Secured_by', 'total_units', 'submission_of_application',
    'Region', 'Security_Type', 'income_type',
    'business_or_commercial', 'open_credit',
    'credit_worthiness', 'construction_type', 'lump_sum_payment',
    ]

    # Keep only available features
    available = [f for f in simple_features if f in df.columns]

    X = df[available]
    y = df['Status']

    print("Feature columns:", available)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

    # Keep only available features
    available = [f for f in simple_features if f in df.columns]

    X = df[available]
    y = df['Status']

    print("Feature columns:", available)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

if __name__ == "__main__":
    df = load_data("Loan_Default.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    X, y = get_features_and_target(df)
    print("Preprocessing complete!")
    # Save preprocessed dataset
    df.to_csv("loan_cleaned.csv", index=False)
    print("Cleaned dataset saved as loan_cleaned.csv!")
