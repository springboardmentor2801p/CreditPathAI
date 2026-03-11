import sqlite3
import pandas as pd

conn = sqlite3.connect("creditpathai.db")
df = pd.read_sql("SELECT * FROM processed_loans", conn)
conn.close()

# ==========================================================
# BOOSTING MODEL EDA TEMPLATE
# Optimized for XGBoost / LightGBM
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1️⃣ BASIC DATA OVERVIEW
# ----------------------------------------------------------

print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum().sort_values(ascending=False))
print("\nUnique Values:\n", df.nunique().sort_values())

# Detect constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("\nConstant Columns:", constant_cols)


# ----------------------------------------------------------
# 2️⃣ TARGET ANALYSIS (MANDATORY)
# ----------------------------------------------------------

target = "Status"   # CHANGE THIS

print("\nTarget Counts:\n", df[target].value_counts())
print("\nTarget Proportion:\n", df[target].value_counts(normalize=True))

plt.figure(figsize=(5,4))
sns.countplot(x=target, data=df)
plt.title("Target Distribution")
plt.show()

# Check if target is scaled
print("\nTarget Unique Values:", df[target].unique())


# ----------------------------------------------------------
# 3️⃣ NUMERIC FEATURES ANALYSIS
# ----------------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove(target)

print("\nNumber of Numeric Features:", len(numeric_cols))

# Check skewness
skewness = df[numeric_cols].skew().sort_values(ascending=False)
print("\nTop Skewed Features:\n", skewness.head(10))

# Plot top 5 skewed features
for col in skewness.head(5).index:
    plt.figure(figsize=(5,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Outlier check (boxplot vs target for top 5)
for col in skewness.head(5).index:
    plt.figure(figsize=(5,4))
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f"{col} vs Target")
    plt.show()


# ----------------------------------------------------------
# 4️⃣ CATEGORICAL FEATURE ANALYSIS
# ----------------------------------------------------------

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# If label-encoded categorical (low unique integers)
for col in numeric_cols:
    if df[col].nunique() < 10:
        categorical_cols.append(col)

categorical_cols = list(set(categorical_cols) - {target})

print("\nCategorical Columns Detected:", categorical_cols)

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

    print(f"\nCross-tab {col} vs Target:")
    print(pd.crosstab(df[col], df[target], normalize="index"))


# ----------------------------------------------------------
# 5️⃣ CORRELATION ANALYSIS
# ----------------------------------------------------------

corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()

print("\nCorrelation with Target:")
print(
    corr_matrix[target]
    .sort_values(ascending=False)
)

# Detect possible leakage (very high correlation)
leakage_features = corr_matrix[target][abs(corr_matrix[target]) > 0.8]
print("\n⚠ Potential Leakage Features:\n", leakage_features)


# ----------------------------------------------------------
# 6️⃣ CLASS IMBALANCE CHECK FOR BOOSTING
# ----------------------------------------------------------

counts = df[target].value_counts()
if len(counts) == 2:
    ratio = counts.max() / counts.min()
    print("\nClass Imbalance Ratio:", round(ratio, 2))

    if ratio > 1.5:
        print("⚠ Dataset is Imbalanced. Use scale_pos_weight.")


# ----------------------------------------------------------
# 7️⃣ FINAL EDA SUMMARY CHECKLIST
# ----------------------------------------------------------

print("\n===== EDA SUMMARY CHECKLIST =====")

print("Total Features:", df.shape[1])
print("Constant Columns:", constant_cols)
print("Highly Skewed Columns (>1):", skewness[abs(skewness) > 1].index.tolist())
print("Highly Correlated Pairs (>0.8):")

high_corr = (
    corr_matrix.abs()
    .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)

print(high_corr[high_corr > 0.8])

print("\nEDA Completed Successfully.")
