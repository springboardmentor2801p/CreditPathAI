# ==========================================================
# XGBOOST TRAINING PIPELINE + CROSS VALIDATION
# ==========================================================

import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------

conn = sqlite3.connect("creditpathai.db")
df = pd.read_sql("SELECT * FROM processed_loans", conn)
conn.close()

print("Initial Shape:", df.shape)


# ----------------------------------------------------------
# 2. BASIC CLEANING
# ----------------------------------------------------------

# Drop constant column
if "year" in df.columns:
    df.drop(columns=["year"], inplace=True)

# Drop correlated duplicates (from EDA)
drop_cols = ["construction_type", "Secured_by"]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Remove leakage features
leakage_features = [
    "Interest_rate_spread",
    "Upfront_charges",
    "rate_of_interest",
    "interest_burden"
]

df.drop(columns=[col for col in leakage_features if col in df.columns], inplace=True)

print("Shape after cleaning:", df.shape)


# ----------------------------------------------------------
# 3. LOG TRANSFORM SKEWED FEATURES
# ----------------------------------------------------------

skew_cols = ["loan_income_ratio", "LTV", "income"]

for col in skew_cols:
    if col in df.columns:
        min_val = df[col].min()

        if min_val <= 0:
            df[col] = np.log1p(df[col] - min_val)
        else:
            df[col] = np.log1p(df[col])

print("Log transformation completed.")


# ----------------------------------------------------------
# 4. DEFINE FEATURES & TARGET
# ----------------------------------------------------------

target = "Status"

X = df.drop(columns=[target])
y = df[target]

print("Feature Count:", X.shape[1])


# ==========================================================
# 5. STRATIFIED K-FOLD CROSS VALIDATION
# ==========================================================

print("\nRunning Stratified K-Fold Cross Validation...\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

    X_train_cv = X.iloc[train_idx]
    X_val_cv = X.iloc[val_idx]

    y_train_cv = y.iloc[train_idx]
    y_val_cv = y.iloc[val_idx]

    neg, pos = y_train_cv.value_counts()
    weight = neg / pos

    model_cv = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42
    )

    model_cv.fit(X_train_cv, y_train_cv)

    preds = model_cv.predict_proba(X_val_cv)[:,1]

    auc = roc_auc_score(y_val_cv, preds)

    cv_scores.append(auc)

    print(f"Fold {fold+1} AUC:", round(auc,4))


print("\nCross Validation Summary")
print("Mean AUC:", round(np.mean(cv_scores),4))
print("Std AUC:", round(np.std(cv_scores),4))


# ==========================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)

print("\nTrain size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)


# ----------------------------------------------------------
# 7. CLASS IMBALANCE HANDLING
# ----------------------------------------------------------

neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos

print("scale_pos_weight:", round(scale_pos_weight,2))


# ----------------------------------------------------------
# 8. MODEL INITIALIZATION
# ----------------------------------------------------------

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42
)


# ----------------------------------------------------------
# 9. TRAIN MODEL
# ----------------------------------------------------------

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print("Best iteration:", model.best_iteration)


# ----------------------------------------------------------
# 10. MODEL EVALUATION
# ----------------------------------------------------------

train_probs = model.predict_proba(
    X_train,
    iteration_range=(0, model.best_iteration)
)[:,1]

test_probs = model.predict_proba(
    X_test,
    iteration_range=(0, model.best_iteration)
)[:,1]


train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)

print("\nTrain ROC-AUC:", round(train_auc,4))
print("Test ROC-AUC:", round(test_auc,4))


# classification metrics

y_pred = (test_probs > 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted 0", "Predicted 1"],
    yticklabels=["Actual 0", "Actual 1"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 11. FEATURE IMPORTANCE
# ----------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8,6))

sns.barplot(
    data=importance_df.head(20),
    x="importance",
    y="feature"
)

plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

import joblib

# Save trained model
joblib.dump(model, "xgboost_model.pkl")

print("Model saved as xgboost_model.pkl")
