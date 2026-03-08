"""
CreditPathAI - End-to-End ML Pipeline
-------------------------------------
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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

df = df.drop_duplicates()

num_cols = [
    "loanAmount",
    "interestRate",
    "monthlyPayment",
    "annualIncome",
    "dtiRatio",
    "revolvingUtilizationRate"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df[(df["loanAmount"] > 0) & (df["annualIncome"] > 0)]

print("Data cleaning completed.")


# =====================================================
# STEP 3: Data Preprocessing
# =====================================================

print("\nSTEP 3: Data Preprocessing")

# Target variable
df["default"] = df["loanStatus"].apply(
    lambda x: 1 if str(x).lower() in ["default", "charged off"] else 0
)

# Drop unnecessary columns
df = df.drop(columns=["loanId", "memberId", "date", "loanStatus"])

# Ordinal encoding
employment_map = {
    "< 1 year": 0,
    "1 year": 1,
    "2-5 years": 3,
    "6-9 years": 7,
    "10+ years": 10
}

df["yearsEmployment"] = df["yearsEmployment"].map(employment_map)
df["yearsEmployment"] = df["yearsEmployment"].fillna(df["yearsEmployment"].median())

# One hot encoding
cat_cols = ["purpose", "grade", "homeOwnership", "term", "residentialState"]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print("Data preprocessing completed.")
print("Final Shape:", df.shape)


# =====================================================
# STEP 4: Train Models
# =====================================================

print("\nSTEP 4: Training Models")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb.fit(X_train, y_train)

# LightGBM
lgb = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

lgb.fit(X_train, y_train)

print("Model training completed.")


# =====================================================
# STEP 5: Model Evaluation
# =====================================================

print("\nSTEP 5: Model Evaluation")

y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_prob_lgb = lgb.predict_proba(X_test)[:, 1]

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
auc_lgb = roc_auc_score(y_test, y_prob_lgb)

print("\nModel Performance (AUC-ROC)")
print("Logistic Regression:", auc_lr)
print("XGBoost:", auc_xgb)
print("LightGBM:", auc_lgb)


# =====================================================
# STEP 6: Save Best Model
# =====================================================

print("\nSTEP 6: Saving Best Model")

joblib.dump(xgb, "backend/model/xgb_model.pkl")

print("Best model saved as: backend/model/xgb_model.pkl")
print("Pipeline completed successfully ✅")
