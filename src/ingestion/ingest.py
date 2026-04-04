import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed/")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sqlite/creditpath.db")

CLEANED_CSV = os.path.join(PROCESSED_DATA_PATH, "cleaned_loan_data.csv")


def load_cleaned_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading cleaned data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def save_to_sqlite(df: pd.DataFrame, table: str = "loan_data"):
    logger.info(f"Connecting to SQLite: {SQLITE_DB_PATH}")
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")

    logger.info(f"Writing to table: '{table}'...")
    df.to_sql(table, engine, if_exists="replace", index=False)

    # Verify
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        cols = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()

    logger.info(f"✅ {count:,} records saved to '{table}'")
    logger.info(f"✅ {len(cols)} columns in table")
    return engine


def verify_db(engine):
    logger.info("Running DB verification...")
    with engine.connect() as conn:
        sample = pd.read_sql("SELECT * FROM loan_data LIMIT 5", conn)
    logger.info(f"Sample rows:\n{sample.head()}")


if __name__ == "__main__":
    # Step 1 — Load cleaned CSV
    df = load_cleaned_data(CLEANED_CSV)

    # Step 2 — Push to SQLite
    engine = save_to_sqlite(df, table="loan_data")

    # Step 3 — Verify
    verify_db(engine)

    logger.info("✅ Ingestion to SQLite complete!")