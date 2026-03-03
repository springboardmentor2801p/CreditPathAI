"""
================================================================================
  CreditPathAI — feature_pipeline.py
  Purpose : Data loading, feature engineering, and sklearn preprocessing.
            Shared by all model scripts — no ML training lives here.
================================================================================
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute        import SimpleImputer


# ─────────────────────────── Constants ──────────────────────────────────────
DB_PATH    = "creditpathai.db"   # overridden by callers via load_data()
TABLE      = "processed_loans"
TARGET_COL = "loanStatus"


# ─────────────────────────── Data Loading ───────────────────────────────────
def load_data(db_path: str, table: str = TABLE) -> pd.DataFrame:
    """Read the feature table from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    print(f"  [data] Loaded '{table}' — {len(df):,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────── Feature Engineering ────────────────────────────
def engineer_features(df: pd.DataFrame,
                      target_col: str = TARGET_COL):
    """
    Split df into X / y and add 5 interaction features on top of
    the 42 already-engineered columns in processed_loans.

    Returns
    -------
    X : pd.DataFrame   (features only, with 5 added interaction cols)
    y : pd.Series      (binary target, int)
    """
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int)

    # 1. Interest-to-DTI burden
    X["interest_dti_burden"] = X["interestRate"] * X["dtiRatio"]

    # 2. Credit stress index  = (delinquencies + derogatory) / (credit history + 1)
    X["credit_stress_index"] = (
        (X["numDelinquency2Years"] + X["numDerogatoryRec"])
        / (X["lengthCreditHistory"] + 1)
    )

    # 3. Payment pressure  = monthly payment / (monthly income)
    X["payment_pressure"] = X["monthlyPayment"] / ((X["annualIncome"] / 12) + 1e-6)

    # 4. Credit depth  = log(totalCreditLines + 1)
    X["credit_depth"] = np.log1p(X["numTotalCreditLines"])

    # 5. Revolving load  = revolvingBalance / annualIncome
    X["revolving_load"] = X["revolvingBalance"] / (X["annualIncome"] + 1)

    added = ["interest_dti_burden", "credit_stress_index",
             "payment_pressure", "credit_depth", "revolving_load"]
    print(f"  [features] Added {len(added)} interaction features: {added}")
    print(f"  [features] Total feature columns: {len(X.columns)}")
    return X, y


# ─────────────────────────── Column Grouping ────────────────────────────────
def get_column_groups(X: pd.DataFrame):
    """
    Partition X columns into:
      binary_cols  – columns with exactly 2 unique values (flags / OHE dummies)
      numeric_cols – everything else (continuous, log-transformed, engineered)
    """
    binary_cols  = [c for c in X.columns if X[c].nunique() == 2]
    numeric_cols = [c for c in X.columns if c not in binary_cols]
    print(f"  [features] Numeric: {len(numeric_cols)}  |  "
          f"Binary: {len(binary_cols)}")
    return numeric_cols, binary_cols


# ─────────────────────────── Preprocessing Pipeline ─────────────────────────
def build_preprocessor(numeric_cols: list, binary_cols: list):
    """
    Build and return a fitted-ready ColumnTransformer:
      • Numeric  → median imputation → RobustScaler
      • Binary   → most-frequent imputation (no scaling needed)
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("bin", binary_transformer,  binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ─────────────────────────── One-shot Helper ────────────────────────────────
def prepare(db_path: str, table: str = TABLE, target_col: str = TARGET_COL):
    """
    Convenience wrapper: load → engineer → group columns → build preprocessor.

    Returns
    -------
    X            : pd.DataFrame
    y            : pd.Series
    preprocessor : ColumnTransformer  (not yet fitted)
    """
    df   = load_data(db_path, table)
    X, y = engineer_features(df, target_col)
    num_cols, bin_cols = get_column_groups(X)
    preprocessor       = build_preprocessor(num_cols, bin_cols)
    return X, y, preprocessor
