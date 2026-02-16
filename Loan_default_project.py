import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("Loan Default.csv")

# Make copy
df_num = df.copy()

# Identify columns
num_cols = df_num.select_dtypes(include=[np.number]).columns
cat_cols = df_num.select_dtypes(include=['object']).columns

# Fill missing values
df_num[num_cols] = df_num[num_cols].fillna(df_num[num_cols].median())
for col in cat_cols:
    df_num[col] = df_num[col].fillna(df_num[col].mode()[0])

# Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df_num[col] = le.fit_transform(df_num[col])

# Scale numeric columns
scaler = StandardScaler()
df_num[num_cols] = scaler.fit_transform(df_num[num_cols])

# Final check
print("Object columns left:", df_num.select_dtypes(include='object').columns)
print("Final shape:", df_num.shape)

# Save fully numeric file
df_num.to_csv(
    r"C:\Users\Asus\Desktop\loan_default_fully_numeric.csv",
    index=False
)

print("✅ Fully numeric dataset saved on Desktop as loan_default_fully_numeric.csv")


