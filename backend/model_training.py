# ===============================
# MODEL TRAINING
# CreditPath AI
# ===============================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import lightgbm as lgb

from data_preprocessing import load_data, clean_data, feature_engineering, get_features_and_target

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_baseline_model(X_train, y_train):
    """Train Logistic Regression baseline model"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Baseline Logistic Regression trained!")
    return model
def train_xgboost(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    best_xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        n_jobs=-1
    )
    best_xgb.fit(X_train, y_train)
    print("XGBoost trained!")
    return best_xgb

def train_lightgbm(X_train, y_train):
    best_lgbm = lgb.LGBMClassifier(
        random_state=42,
        verbose=-1,
        class_weight='balanced',
        n_estimators=300,
        num_leaves=127,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1
    )
    best_lgbm.fit(X_train, y_train)
    print("LightGBM trained!")
    return best_lgbm

def evaluate_model(model, X_test, y_test, name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return auc

def save_model(model, scaler, model_path="creditpath_model.pkl", scaler_path="creditpath_scaler.pkl"):
    """Save trained model and scaler"""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    df = load_data("Loan_Default.csv")
    df = clean_data(df)
    df = feature_engineering(df)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Train models
    baseline = train_baseline_model(X_train_scaled, y_train)
    auc_baseline = evaluate_model(baseline, X_test_scaled, y_test, "Logistic Regression")

    best_xgb = train_xgboost(X_train, y_train)
    auc_xgb = evaluate_model(best_xgb, X_test, y_test, "XGBoost")

    best_lgbm = train_lightgbm(X_train, y_train)
    auc_lgbm = evaluate_model(best_lgbm, X_test, y_test, "LightGBM")

    # Save best model
    scores = {
        "LogisticRegression": auc_baseline,
        "XGBoost": auc_xgb,
        "LightGBM": auc_lgbm
    }
    best_name = max(scores, key=scores.get)
    best_model = {
        "LogisticRegression": baseline,
        "XGBoost": best_xgb,
        "LightGBM": best_lgbm
    }[best_name]

    print(f"\nBest Model: {best_name} → AUC-ROC: {scores[best_name]:.4f}")
    save_model(best_model, scaler)
    print("Training complete!")