import pandas as pd
from sqlalchemy import create_engine

USERNAME = "postgres"
PASSWORD = "12345"   # replace with your real password
HOST = "localhost"
PORT = "5432"
DATABASE = "creditpathai"

engine = create_engine(
    f'postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
)

try:
    df = pd.read_sql("SELECT * FROM cleaned_loan_data", engine)
    print("Data Loaded Successfully!")

    # -------------------------------
    # Feature Engineering
    # -------------------------------

    # Loan to income ratio
    df['loan_to_income_ratio'] = df['num_loans_taken'] / df['monthly_income']

    # Expense ratio
    df['expense_ratio'] = df['monthly_expense'] / df['monthly_income']

    # Store new table
    df.to_sql("feature_engineered_data", engine, if_exists="replace", index=False)

    print("Feature engineered data stored successfully!")

except Exception as e:
    print("Error:", e)