# ingestion.py
import pandas as pd
import os

# Path to data folder
data_folder = os.path.join("..", "data")

# List of CSV files
files = ["data_01.csv", "data_02.csv", "data_for_training.csv"]

for file in files:
    file_path = os.path.join(data_folder, file)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"{file} loaded successfully! Shape: {df.shape}")
        print(df.head(), "\n")
    else:
        print(f"Error: {file} not found in {data_folder}")
