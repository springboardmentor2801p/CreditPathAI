import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import json

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
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

# Output dirs
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)


def load_data() -> pd.DataFrame:
    logger.info("Loading engineered data from SQLite...")
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df = pd.read_sql("SELECT * FROM loan_engineered", engine)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """Split into X and y, drop non-feature columns."""
    logger.info("Preparing features and target...")

    # Drop columns not useful for modeling
    drop_cols = ['id', 'year']
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Target
    target = 'status'
    X = df.drop(columns=[target])
    y = df[target]

    logger.info(f"Features: {X.shape[1]} | Target distribution:")
    logger.info(f"  No Default (0): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    logger.info(f"  Default    (1): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

    return X, y


def split_data(X, y):
    """Train/test split with stratification."""
    logger.info(f"Splitting data: {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    """Handle class imbalance using SMOTE."""
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE - Train size: {X_res.shape[0]:,}")
    logger.info(f"  No Default (0): {(y_res==0).sum():,}")
    logger.info(f"  Default    (1): {(y_res==1).sum():,}")
    return X_res, y_res


def evaluate_model(model, X_test, y_test) -> dict:
    """Generate all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info("Model Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info("\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")

    logger.info("Confusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with MLflow tracking."""

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    params = {
        "model": "LogisticRegression",
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
        "smote": True,
        "cv_folds": CV_FOLDS
    }

    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        logger.info("Training Logistic Regression...")

        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=params['C'],
                max_iter=params['max_iter'],
                solver=params['solver'],
                random_state=params['random_state'],
                class_weight='balanced'
            ))
        ])

        # Cross validation
        logger.info(f"Running {CV_FOLDS}-fold cross validation...")
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE),
            scoring='roc_auc'
        )
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Final training
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        mlflow.sklearn.log_model(pipeline, "logistic_regression_model")

        # Save model locally
        model_path = "data/models/logistic_regression.pkl"
        joblib.dump(pipeline, model_path)
        logger.info(f"Model saved to: {model_path}")

        # Save metrics report
        report_path = "data/reports/logistic_regression_metrics.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to: {report_path}")

        logger.info("MLflow run complete!")
        return pipeline, metrics


if __name__ == "__main__":
    # Load data
    df = load_data()

    # Prepare
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Handle imbalance
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Train
    model, metrics = train_logistic_regression(
        X_train_res, y_train_res,
        X_test, y_test
    )

    logger.info("Baseline model training complete!")
    logger.info(f"Final ROC-AUC: {metrics['roc_auc']}")