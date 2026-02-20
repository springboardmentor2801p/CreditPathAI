import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading datasets...")


loan = pd.read_csv("Loan.csv")
borrower = pd.read_csv("Borrower.csv")

print("Loan shape:", loan.shape)
print("Borrower shape:", borrower.shape)


data = pd.merge(loan, borrower, on="memberId", how="inner")

print("Merged shape:", data.shape)

data = data.fillna(data.median(numeric_only=True))

data['loanStatus'] = data['loanStatus'].map({
    'Current': 0,
    'Default': 1
})

data = pd.get_dummies(data, drop_first=True)

scaler = StandardScaler()

numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

data = data.drop(columns=['loanId', 'memberId'], errors='ignore')


data.to_csv("Merged_Preprocessed_Data.csv", index=False)

print("✅ Merging and Preprocessing Completed Successfully!")
