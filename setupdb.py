import sqlite3
import pandas as pd

loan_df = pd.read_csv('D:\jai\Python-Workspace\Credit-Path-AI\data\Loan.txt', sep = '\t')
borrower_df = pd.read_csv('D:\jai\Python-Workspace\Credit-Path-AI\data\Borrower.txt', sep='\t')

conn = sqlite3.connect("D:\jai\Python-Workspace\Credit-Path-AI\database\creditpathai.db")

loan_df.to_sql("loans", conn, if_exists="replace", index=False)
borrower_df.to_sql("borrowers", conn, if_exists="replace", index=False)

conn.close()

print("CSV Successfully converted to SQLite")