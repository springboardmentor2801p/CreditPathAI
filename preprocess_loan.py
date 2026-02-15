import pandas as pd

# Load dataset
df = pd.read_csv("Loan.csv")

# Remove duplicates
df = df.drop_duplicates()

# Separate numeric & categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill numeric missing values with mean
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical missing values with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned dataset
df.to_csv("Loan_cleaned_v2.csv", index=False)

print("✅ Numerical and categorical missing values handled successfully")
