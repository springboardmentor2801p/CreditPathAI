import pandas as pd
import os

# Load dataset
file_path = os.path.join("..", "Data", "cleaned_training_data.csv")
df = pd.read_csv(file_path)

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())