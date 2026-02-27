# ================================
# 1️⃣ Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# ================================
# 2️⃣ Load Dataset (Choose one)
# ================================

# Option 1: Load from CSV
# df = pd.read_csv("loan_data.csv")

# Option 2: Load from SQLite Database
conn = sqlite3.connect("credit.db")
df = pd.read_sql("SELECT * FROM loan_table", conn)

# ================================
# 3️⃣ Understand Data
# ================================
print("Shape of dataset:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# ================================
# 4️⃣ Handle Missing Values
# ================================
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Fill numeric columns with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# ================================
# 5️⃣ Analyze Categorical Variables
# ================================
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.xticks(rotation=45)
    plt.title(f"Count Plot of {col}")
    plt.show()

# ================================
# 6️⃣ Correlation Analysis (Heatmap)
# ================================
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 7️⃣ Outlier Detection (Boxplots)
# ================================
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
