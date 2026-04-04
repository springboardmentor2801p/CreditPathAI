import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/ingest_raw.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/Loan_default.csv")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sqlite/creditpath.db")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/")


def load_raw_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw CSV from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning raw data...")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicates removed: {before - len(df)}")

    # Convert numeric columns
    numeric_cols = [
        'loan_amount', 'rate_of_interest', 'interest_rate_spread',
        'upfront_charges', 'term', 'property_value', 'income',
        'credit_score', 'ltv', 'dtir1'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    logger.info(f"Cleaned shape: {df.shape}")
    return df


def save_to_sqlite(df: pd.DataFrame, table: str = "loan_raw"):
    logger.info(f"Saving to SQLite table: {table}")
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df.to_sql(table, engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    logger.info(f"Saved {count:,} records to '{table}'")


def save_processed_csv(df: pd.DataFrame):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out = os.path.join(PROCESSED_DATA_PATH, "cleaned_raw.csv")
    df.to_csv(out, index=False)
    logger.info(f"Saved cleaned CSV to: {out}")


if __name__ == "__main__":
    df = load_raw_data(RAW_DATA_PATH)
    df = clean_raw_data(df)
    save_to_sqlite(df, table="loan_raw")
    save_processed_csv(df)
    logger.info("✅ Raw ingestion complete!")