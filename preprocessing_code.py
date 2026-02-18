import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# Load dataset
df = pd.read_csv("merged_borrower_loan.csv")

# Remove duplicates
df = df.drop_duplicates()

# Drop ID column
if 'memberId' in df.columns:
    df = df.drop(columns=['memberId'])

# Drop columns with >50% missing values
missing_percent = df.isnull().mean()
df = df.loc[:, missing_percent < 0.5]

# Handle missing values
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Label encoding
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Remove low variance features
selector = VarianceThreshold(threshold=0.01)
df = pd.DataFrame(selector.fit_transform(df),
                  columns=df.columns[selector.get_support()])

# Remove highly correlated features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
df = df.drop(columns=to_drop)

# Separate target
y = df["loanStatus"]
X = df.drop(columns=["loanStatus"])

# Scale only features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Combine back
final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

# Save
final_df.to_csv("preprocessed_dataset.csv", index=False)
