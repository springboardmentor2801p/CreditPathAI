

# 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 2. Loading the Merged Dataset

df = pd.read_csv("final_preprocessed_dataset.csv")


# 3. Removing unnecessary column
# 'grade' column had no useful values, so it is removed
df.drop(columns=["grade"], inplace=True)


# 4. Convert Loan Term to Numeric
# Extract numeric value from strings like "36 months"
df["term"] = df["term"].str.extract("(\d+)")
df["term"] = pd.to_numeric(df["term"])

# Fill missing values with median
df["term"] = df["term"].fillna(df["term"].median())


# 5. Convert Employment Length to Numeric
# Convert employment categories into numeric years

def convert_employment(x):
    if pd.isna(x):
        return np.nan
    elif "10+" in str(x):
        return 10
    elif "<" in str(x):
        return 0
    elif "-" in str(x):
        return int(str(x).split("-")[0])
    else:
        return int(str(x).split()[0])

df["yearsEmployment"] = df["yearsEmployment"].apply(convert_employment)

# Fill missing employment values with median
df["yearsEmployment"] = df["yearsEmployment"].fillna(
    df["yearsEmployment"].median()
)


# 6. Feature Engineering from Date Column

# Convert to datetime format
df["date"] = pd.to_datetime(df["date"])

# Extract useful features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Drop original date column
df.drop(columns=["date"], inplace=True)


# 7. Encode Target Variable
# Convert loanStatus:
# Current -> 0
# Default -> 1
df["loanStatus"] = df["loanStatus"].map({
    "Current": 0,
    "Default": 1
})


# 8. Handle String Missing Values
# Replace "NM" entries with NaN
df.replace("NM", np.nan, inplace=True)


# 9. Convert Boolean Columns to Integer

# True -> 1, False -> 0
bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)


# 10. Handle Remaining Missing Values
# Fill remaining numeric missing values using median
df.fillna(df.median(numeric_only=True), inplace=True)


# 11. Feature Scaling
# Standardize features for ML models

scaler = StandardScaler()

X = df.drop("loanStatus", axis=1)
y = df["loanStatus"]

X_scaled = scaler.fit_transform(X)

# Recreate DataFrame after scaling
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["loanStatus"] = y.values


# 12. Save Final Preprocessed Dataset
# ----------------------------------------------------------
df_scaled.to_csv("final_preprocessed_dataset.csv", index=False)

print("Data preprocessing completed successfully.")
