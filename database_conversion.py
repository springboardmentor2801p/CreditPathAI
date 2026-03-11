import pandas as pd
import sqlite3

# Connect (or create) SQLite database
conn = sqlite3.connect("creditpathai.db")


# 1. Load Raw Dataset

raw_df = pd.read_csv("data/raw/Loan_Default.csv")

raw_df.to_sql(
    "raw_loans",    # table name
    conn,
    if_exists="replace",
    index=False
)

print("Raw dataset successfully stored in SQLite.")


# 2. Load Processed Dataset

processed_df = pd.read_csv("data/processed/preprocessed.csv")

processed_df.to_sql(
    "processed_loans",    # table name
    conn,
    if_exists="replace",
    index=False
)

print("Processed dataset successfully stored in SQLite.")

print("Database creation completed successfully!")
# To Check whether the tables are created or not :
tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

print(tables)

# Close connection
conn.close()