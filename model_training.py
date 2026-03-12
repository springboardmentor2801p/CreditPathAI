# CreditPathAI - Model Training (Logistic Regression, XGBoost, LightGBM)

# This script trains three machine learning models to predict loan default risk.
# Models used:
# 1. Logistic Regression (Baseline model)
# 2. XGBoost (Advanced boosting model)
# 3. LightGBM (Gradient boosting framework)

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. Load data from SQLite database

conn = sqlite3.connect("creditpathai.db")

df = pd.read_sql("SELECT * FROM processed_loans", conn)

conn.close()

# 2. Define features and target variable

# Target variable
y = df["loanStatus"]

# Feature set
X = df.drop(columns=["loanStatus"])

# 3. Train-test split

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Logistic Regression (Baseline Model)

pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline_lr.fit(X_train, y_train)

# Predict probabilities
lr_probs = pipeline_lr.predict_proba(X_test)[:,1]

# Model evaluation
lr_auc = roc_auc_score(y_test, lr_probs)
lr_pred = pipeline_lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Results")
print("AUC-ROC:", lr_auc)
print("Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_pred))


# 5. XGBoost Model

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

pipeline_xgb = Pipeline([
    ("model", xgb_model)
])

pipeline_xgb.fit(X_train, y_train)

# Predict probabilities
xgb_probs = pipeline_xgb.predict_proba(X_test)[:,1]

# Model evaluation
xgb_auc = roc_auc_score(y_test, xgb_probs)
xgb_pred = pipeline_xgb.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\nXGBoost Results")
print("AUC-ROC:", xgb_auc)
print("Accuracy:", xgb_accuracy)
print(classification_report(y_test, xgb_pred))

# 6. LightGBM Model

lgb_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

pipeline_lgb = Pipeline([
    ("model", lgb_model)
])

pipeline_lgb.fit(X_train, y_train)

# Predict probabilities
lgb_probs = pipeline_lgb.predict_proba(X_test)[:,1]

# Model evaluation
lgb_auc = roc_auc_score(y_test, lgb_probs)
lgb_pred = pipeline_lgb.predict(X_test)
lgb_accuracy = accuracy_score(y_test, lgb_pred)

print("\nLightGBM Results")
print("AUC-ROC:", lgb_auc)
print("Accuracy:", lgb_accuracy)
print(classification_report(y_test, lgb_pred))

# 7. Model Comparison

print("\nModel Comparison Summary")

print("Logistic Regression AUC:", lr_auc)
print("XGBoost AUC:", xgb_auc)
print("LightGBM AUC:", lgb_auc)

print("\nLogistic Regression Accuracy:", lr_accuracy)
print("XGBoost Accuracy:", xgb_accuracy)
print("LightGBM Accuracy:", lgb_accuracy)

