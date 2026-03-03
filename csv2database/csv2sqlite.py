import sqlite3
import pandas as pd

# File paths and table names
csv1_path = 'raw_loans.csv'  # Replace with your first CSV file path
csv2_path = 'processed_loans.csv'  # Replace with your second CSV file path

table1_name = 'raw_loans'
table2_name = 'processed_loans'

db_path = 'creditpathai.db'

# Connect to SQLite database (creates if doesn't exist)
conn = sqlite3.connect(db_path)

try:
    # Read CSV files into pandas DataFrames
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Import to SQLite tables (if_exists='replace' overwrites, 'append' adds data)
    df1.to_sql(table1_name, conn, if_exists='replace', index=False)
    df2.to_sql(table2_name, conn, if_exists='replace', index=False)
    
    print(f"Successfully imported {len(df1)} rows to {table1_name}")
    print(f"Successfully imported {len(df2)} rows to {table2_name}")
    
finally:
    # Close connection
    conn.close()
