import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import json

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/advanced_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sqlite/creditpath.db")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlflow_runs/")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "loan_default_prediction")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
CV_FOLDS = int(os.getenv("CV_FOLDS", 5))

os.makedirs("data/models", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)


def load_data() -> pd.DataFrame:
    logger.info("Loading engineered data from SQLite...")
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df = pd.read_sql("SELECT * FROM loan_engineered", engine)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    drop_cols = [c for c in ['id', 'year'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    target = 'status'
    X = df.drop(columns=[target])
    y = df[target]
    logger.info(f"Features: {X.shape[1]} | No Default: {(y==0).sum():,} | Default: {(y==1).sum():,}")
    return X, y


def split_and_smote(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE - Train size: {X_train_res.shape[0]:,}")
    return X_train_res, X_test, y_train_res, y_test


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {name} Results:")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            logger.info(f"  {k}: {v}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    return metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    params = {
        "model": "XGBoost",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    with mlflow.start_run(run_name="XGBoost"):
        logger.info("Training XGBoost...")

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            eval_metric=params["eval_metric"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"]
        )

        # CV
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        logger.info(f"XGBoost CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        model.fit(X_train, y_train)
        metrics = evaluate_model("XGBoost", model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.sklearn.log_model(model, "xgboost_model")

        joblib.dump(model, "data/models/xgboost_model.pkl")
        with open("data/reports/xgboost_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("XGBoost training complete!")
        return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    params = {
        "model": "LightGBM",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1
    }

    with mlflow.start_run(run_name="LightGBM"):
        logger.info("Training LightGBM...")

        model = LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"],
            verbose=params["verbose"]
        )

        # CV
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        logger.info(f"LightGBM CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        model.fit(X_train, y_train)
        metrics = evaluate_model("LightGBM", model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.sklearn.log_model(model, "lightgbm_model")

        joblib.dump(model, "data/models/lightgbm_model.pkl")
        with open("data/reports/lightgbm_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info("LightGBM training complete!")
        return model, metrics


def compare_models(lr_metrics, xgb_metrics, lgbm_metrics):
    logger.info("\n" + "="*60)
    logger.info("  MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Metric':<15} {'LogReg':>10} {'XGBoost':>10} {'LightGBM':>10}")
    logger.info("-"*50)

    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        lr_val = lr_metrics.get(metric, "-")
        xgb_val = xgb_metrics.get(metric, "-")
        lgbm_val = lgbm_metrics.get(metric, "-")
        logger.info(f"{metric:<15} {lr_val:>10} {xgb_val:>10} {lgbm_val:>10}")

    # Save comparison
    comparison = {
        "LogisticRegression": lr_metrics,
        "XGBoost": xgb_metrics,
        "LightGBM": lgbm_metrics
    }
    with open("data/reports/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)
    logger.info("\nComparison saved to: data/reports/model_comparison.json")


if __name__ == "__main__":
    # Load baseline metrics
    with open("data/reports/logistic_regression_metrics.json") as f:
        lr_metrics = json.load(f)

    # Load data
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_and_smote(X, y)

    # Train advanced models
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    lgbm_model, lgbm_metrics = train_lightgbm(X_train, y_train, X_test, y_test)

    # Compare all 3
    compare_models(lr_metrics, xgb_metrics, lgbm_metrics)

    logger.info("Advanced model training complete!")