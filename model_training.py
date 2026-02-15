import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned.csv")

print(df.shape)

# Duplicate check
print("Duplicate rows:", df.duplicated().sum())
