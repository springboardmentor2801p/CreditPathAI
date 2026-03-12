import pandas as pd
import sqlite3

# Step 1: Connect to (or create) SQLite database
conn = sqlite3.connect("creditpathai.db")

# Step 2: Load raw dataset
df_raw = pd.read_csv("merged_borrower_loan.csv")

# Step 3: Store raw dataset as table
df_raw.to_sql("raw_loans", conn, if_exists="replace", index=False)

# Step 4: Load processed dataset
df_processed = pd.read_csv("preprocessed_dataset.csv")

# Step 5: Store processed dataset as table
df_processed.to_sql("processed_loans", conn, if_exists="replace", index=False)

# Step 6: Verify tables inside database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in database:", cursor.fetchall())

# Step 7: Close connection
conn.close()

print("Database successfully created and both tables added!")
