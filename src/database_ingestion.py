import pandas as pd
import sqlite3
import os

# Database file path
db_path = os.path.join("..", "Data", "creditpathai.db")

# CSV file path
csv_path = os.path.join("..", "Data", "cleaned_training_data.csv")

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Load CSV
df = pd.read_csv(csv_path)

# Store into database table
df.to_sql("loan_data", conn, if_exists="replace", index=False)

print("CSV successfully stored in SQLite Database!")

conn.close()
