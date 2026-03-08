import pandas as pd
from sqlalchemy import create_engine

# -------------------------------
# PostgreSQL Connection Details
# -------------------------------
USERNAME = "postgres"
PASSWORD = "12345"          # change if needed
HOST = "localhost"
PORT = "5432"
DATABASE = "creditpathai"

# -------------------------------
# Create Database Connection
# -------------------------------
engine = create_engine(
    f'postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
)

try:
    # -------------------------------
    # Load Raw Dataset
    # -------------------------------
    raw_data = pd.read_csv("../Data/data_for_training.csv")

    # Store Raw Data into PostgreSQL
    raw_data.to_sql(
        name="raw_loan_data",
        con=engine,
        if_exists="replace",
        index=False
    )
    print("Raw data stored successfully!")

    # -------------------------------
    # Load Cleaned Dataset
    # -------------------------------
    cleaned_data = pd.read_csv("../Data/cleaned_training_data.csv")

    # Store Cleaned Data into PostgreSQL
    cleaned_data.to_sql(
        name="cleaned_loan_data",
        con=engine,
        if_exists="replace",
        index=False
    )
    print("Cleaned data stored successfully!")

except Exception as e:
    print("Error during ingestion:", e)