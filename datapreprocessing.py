"""
CreditPathAI - Data Preprocessing Pipeline
------------------------------------------
STEP 1: Data Ingestion
STEP 2: Data Cleaning
STEP 3: Data Preprocessing
STEP 4: Save Final Dataset
"""

import pandas as pd
import numpy as np


# =====================================================
# STEP 1: Data Ingestion
# =====================================================

print("STEP 1: Data Ingestion")

loan = pd.read_csv("backend/data/Loan.csv")
borrower = pd.read_csv("backend/data/Borrower.csv")

df = pd.merge(loan, borrower, on="memberId", how="inner")

print("Datasets merged successfully.")
print("Initial Shape:", df.shape)


# =====================================================
# STEP 2: Data Cleaning
# =====================================================

print("\nSTEP 2: Data Cleaning")

# Remove duplicates
df = df.drop_duplicates()

# Fix numeric data types
num_fix_cols = [
    "loanAmount",
    "interestRate",
    "monthlyPayment",
    "annualIncome",
    "dtiRatio",
    "revolvingUtilizationRate"
]

for col in num_fix_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle missing values
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Standardize categorical text
text_cols = ["purpose", "homeOwnership", "grade"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].str.lower().str.strip()

# Remove invalid records
df = df[(df["loanAmount"] > 0) & (df["annualIncome"] > 0)]

print("Data cleaning completed.")


# =====================================================
# STEP 3: Data Preprocessing
# =====================================================

print("\nSTEP 3: Data Preprocessing")

# Target variable creation
if "loanStatus" in df.columns:
    df["default"] = df["loanStatus"].apply(
        lambda x: 1 if str(x).lower() == "default" else 0
    )

# Drop non-predictive columns
cols_to_drop = ["loanId", "memberId", "date"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Ordinal encoding (yearsEmployment)
employment_map = {
    "< 1 year": 0,
    "1 year": 1,
    "2-5 years": 3,
    "6-9 years": 7,
    "10+ years": 10
}

if "yearsEmployment" in df.columns:
    df["yearsEmployment"] = df["yearsEmployment"].map(employment_map)
    df["yearsEmployment"] = df["yearsEmployment"].fillna(
        df["yearsEmployment"].median()
    )

# One-hot encode categorical variables
cat_cols = ["purpose", "grade", "homeOwnership", "term", "residentialState"]

df = pd.get_dummies(
    df,
    columns=[c for c in cat_cols if c in df.columns],
    drop_first=True
)

print("Data preprocessing completed.")


# =====================================================
# STEP 4: Save Final Dataset
# =====================================================

output_path = "backend/data/final_preprocessed_loan_data.csv"
df.to_csv(output_path, index=False)

print("Preprocessing Complete ✅")
print(f"Final dataset saved as: {output_path}")
print("Final Shape:", df.shape)
print(df.head())
