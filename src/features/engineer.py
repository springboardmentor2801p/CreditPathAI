import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sqlite/creditpath.db")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/")


def load_from_sqlite() -> pd.DataFrame:
    logger.info("Loading data from SQLite...")
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df = pd.read_sql("SELECT * FROM loan_data", engine)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def create_ltv_band(df: pd.DataFrame) -> pd.DataFrame:
    """Loan To Value ratio bands — higher LTV = higher risk."""
    logger.info("Creating LTV bands...")
    bins = [0, 60, 75, 90, 100, float('inf')]
    labels = [0, 1, 2, 3, 4]  # 0=lowest risk, 4=highest risk
    df['ltv_band'] = pd.cut(df['ltv'], bins=bins, labels=labels)
    df['ltv_band'] = df['ltv_band'].astype(int)
    return df


def create_income_loan_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of loan amount to income — affordability indicator."""
    logger.info("Creating income to loan ratio...")
    df['income_loan_ratio'] = df['loan_amount'] / (df['income'] + 1)
    df['income_loan_ratio'] = df['income_loan_ratio'].replace(
        [np.inf, -np.inf], 0
    ).fillna(0)
    return df


def create_credit_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Simple rule-based credit risk score (0-100)."""
    logger.info("Creating credit risk score...")

    score = pd.Series(50.0, index=df.index)  # base score

    # Credit score impact (-20 to +20)
    if 'credit_score' in df.columns:
        score += (df['credit_score'] - df['credit_score'].median()) / \
                  df['credit_score'].std() * 10

    # LTV impact — high LTV increases risk
    if 'ltv' in df.columns:
        score -= (df['ltv'] - 70) * 0.3

    # Income loan ratio — high ratio increases risk
    if 'income_loan_ratio' in df.columns:
        score -= df['income_loan_ratio'] * 2

    # DTI ratio impact
    if 'dtir1' in df.columns:
        score -= (df['dtir1'] - 40) * 0.2

    # Clip to 0-100
    df['credit_risk_score'] = score.clip(0, 100).round(2)
    return df


def create_interest_burden(df: pd.DataFrame) -> pd.DataFrame:
    """Total interest burden over loan term."""
    logger.info("Creating interest burden...")
    if 'rate_of_interest' in df.columns and 'term' in df.columns:
        df['interest_burden'] = (
            df['rate_of_interest'] * df['term'] * df['loan_amount']
        ) / 100
        df['interest_burden'] = df['interest_burden'].fillna(0)
    return df


def create_property_loan_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Gap between property value and loan amount."""
    logger.info("Creating property loan gap...")
    if 'property_value' in df.columns:
        df['property_loan_gap'] = df['property_value'] - df['loan_amount']
    return df


def create_high_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag for high risk applicants."""
    logger.info("Creating high risk flag...")
    df['high_risk_flag'] = (
        (df['ltv'] > 90) |
        (df['credit_risk_score'] < 30) |
        (df['income_loan_ratio'] > 5)
    ).astype(int)
    return df


def save_engineered_data(df: pd.DataFrame):
    """Save to both CSV and SQLite."""
    # Save CSV
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_PATH, "engineered_loan_data.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved engineered CSV to: {out_path}")

    # Save to SQLite
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df.to_sql("loan_engineered", engine, if_exists="replace", index=False)
    logger.info("Saved to SQLite table: 'loan_engineered'")


if __name__ == "__main__":
    # Load
    df = load_from_sqlite()

    # Engineer features
    df = create_ltv_band(df)
    df = create_income_loan_ratio(df)
    df = create_credit_risk_score(df)
    df = create_interest_burden(df)
    df = create_property_loan_gap(df)
    df = create_high_risk_flag(df)

    # Summary
    logger.info(f"New features added: ltv_band, income_loan_ratio, "
                f"credit_risk_score, interest_burden, "
                f"property_loan_gap, high_risk_flag")
    logger.info(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Save
    save_engineered_data(df)
    logger.info("Feature engineering complete!")