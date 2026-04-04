import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/Loan_default.csv")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/")


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Standardizing column names...")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicates removed: {before - len(df)}")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Fixing data types...")
    numeric_cols = [
        'loan_amount', 'rate_of_interest', 'interest_rate_spread',
        'upfront_charges', 'term', 'property_value', 'income',
        'credit_score', 'ltv', 'dtir1'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Imputing missing values...")

    # Numeric → median
    for col in df.select_dtypes(include=np.number).columns:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            logger.info(f"  {col}: filled {missing} missing with median")

    # Categorical → mode
    for col in df.select_dtypes(include='object').columns:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            logger.info(f"  {col}: filled {missing} missing with mode")

    return df

def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that would not be available at loan application time."""
    leaky_cols = ['interest_rate_spread', 'upfront_charges', 'rate_of_interest']
    cols_to_drop = [c for c in leaky_cols if c in df.columns]
    logger.info(f"Dropping leaky columns: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    return df
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding categorical columns...")
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    # Remove target if present
    if 'status' in cat_cols:
        cat_cols.remove('status')

    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
        logger.info(f"  Encoded: {col}")

    return df


def save_processed(df: pd.DataFrame):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_PATH, "cleaned_loan_data.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"✅ Preprocessed CSV saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    df = standardize_columns(df)
    df = remove_duplicates(df)
    df = fix_dtypes(df)
    df = impute_missing(df)
    df = drop_leaky_columns(df)
    df = encode_categoricals(df)
    save_processed(df)
    logger.info("✅ Preprocessing complete!")