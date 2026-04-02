import sqlite3
import pandas as pd


def create_database():

    # Load CSV
    df = pd.read_csv("data/raw/Loan_Default.csv")

    # Connect to SQLite
    conn = sqlite3.connect("data/loan.db")

    # Save to DB
    df.to_sql("loan_data", conn, if_exists="replace", index=False)

    conn.close()

    print("✅ Database created successfully!")


def load_from_db():

    conn = sqlite3.connect("data/loan.db")

    df = pd.read_sql("SELECT * FROM loan_data", conn)

    conn.close()

    print("Data loaded from DB:", df.shape)

    return df


if __name__ == "__main__":
    create_database()
    df = load_from_db()
    print(df.head())