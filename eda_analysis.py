
# CreditPathAI - Exploratory Data Analysis (EDA)


import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data from SQLite database

# Connect to the database
conn = sqlite3.connect("creditpathai.db")

# Load the processed loan data
df = pd.read_sql("SELECT * FROM processed_loans", conn)

# Close the database connection
conn.close()

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 2. Check dataset size

print("\nDataset Shape (rows, columns):")
print(df.shape)

# 3. Inspect dataset structure

print("\nDataset Info:")
df.info()

# 4. Statistical summary of numerical features

print("\nStatistical Summary:")
print(df.describe())

# 5. View column names

print("\nColumn Names:")
print(df.columns)

# 6. Target variable distribution (Loan Status)

print("\nLoan Status Counts:")
print(df['loanStatus'].value_counts())

print("\nLoan Status Proportion:")
print(df['loanStatus'].value_counts(normalize=True))

# 7. Visualize loan status distribution

plt.figure(figsize=(6,4))
df['loanStatus'].value_counts().plot(kind='bar')

plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")

plt.show()

# 8. Distribution of numerical features

df.hist(figsize=(12,10))
plt.suptitle("Distribution of Numerical Features")
plt.show()

# 9. Employment distribution

plt.figure(figsize=(8,5))

sns.countplot(x='yearsEmployment', data=df)

plt.title("Employment Length Distribution")
plt.xlabel("Years of Employment")
plt.ylabel("Count")

plt.xticks(rotation=45)
plt.show()

# 10. Correlation heatmap

plt.figure(figsize=(12,10))

corr_matrix = df.corr()

sns.heatmap(corr_matrix, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()

# 11. Feature comparison with loan default

# Loan Amount vs Default
plt.figure(figsize=(6,4))
sns.boxplot(x='loanStatus', y='loanAmount', data=df)
plt.title("Loan Amount vs Loan Status")
plt.show()

# Annual Income vs Default
plt.figure(figsize=(6,4))
sns.boxplot(x='loanStatus', y='annualIncome', data=df)
plt.title("Annual Income vs Loan Status")
plt.show()

# Revolving Utilization vs Default
plt.figure(figsize=(6,4))
sns.boxplot(x='loanStatus', y='revolvingUtilizationRate', data=df)
plt.title("Revolving Utilization vs Loan Status")
plt.show()

# Debt to Income Ratio vs Default
plt.figure(figsize=(6,4))
sns.boxplot(x='loanStatus', y='dtiRatio', data=df)
plt.title("Debt-to-Income Ratio vs Loan Status")
plt.show()

# 12. Outlier detection in loan amount

plt.figure(figsize=(6,4))

sns.boxplot(x=df['loanAmount'])

plt.title("Outliers in Loan Amount")

plt.show()

