import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -----------------------------
# Database Connection
# -----------------------------
USERNAME = "postgres"
PASSWORD = "12345"  # <-- put your real password
HOST = "localhost"
PORT = "5432"
DATABASE = "creditpathai"

engine = create_engine(
    f'postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_sql("SELECT * FROM feature_engineered_data", engine)

X = df.drop("target", axis=1)

print("Feature Columns:", X.columns)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 1️⃣ XGBOOST MODEL
# =========================================================
print("Training XGBoost...")

xgb = XGBClassifier(eval_metric='logloss')

xgb.fit(X_train, y_train)

xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_probs)

print("XGBoost AUC:", xgb_auc)


# =========================================================
# 2️⃣ LIGHTGBM MODEL
# =========================================================
print("\nTraining LightGBM...")

lgb = LGBMClassifier()

lgb.fit(X_train, y_train)

lgb_probs = lgb.predict_proba(X_test)[:, 1]
lgb_auc = roc_auc_score(y_test, lgb_probs)

print("LightGBM AUC:", lgb_auc)


# =========================================================
# 3️⃣ HYPERPARAMETER TUNING (XGBOOST)
# =========================================================
print("\nTuning XGBoost...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss'),
    param_grid,
    cv=3,
    scoring="roc_auc"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_probs = best_model.predict_proba(X_test)[:, 1]
best_auc = roc_auc_score(y_test, best_probs)

print("Best Parameters:", grid.best_params_)
print("Tuned XGBoost AUC:", best_auc)


# =========================================================
# 4️⃣ FINAL COMPARISON
# =========================================================
print("\nModel Comparison:")
print("Baseline Logistic Regression AUC: 0.998")
print("XGBoost AUC:", xgb_auc)
print("LightGBM AUC:", lgb_auc)
print("Tuned XGBoost AUC:", best_auc)
# =========================================================
# 5️⃣ SAVE FINAL MODEL
# =========================================================
import joblib

joblib.dump(best_model, "final_model.pkl")

print("Final model saved successfully!")