import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json
import logging
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/sqlite/creditpath.db")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
REPORTS_PATH = "data/reports/shap/"
os.makedirs(REPORTS_PATH, exist_ok=True)


def load_data():
    logger.info("Loading engineered data...")
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")
    df = pd.read_sql("SELECT * FROM loan_engineered", engine)
    drop_cols = [c for c in ['id', 'year'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    X = df.drop(columns=['status'])
    y = df['status']
    return X, y


def get_test_set(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def plot_shap_summary(model, X_test, model_name):
    logger.info(f"Computing SHAP values for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar",
                      show=False, max_display=15)
    plt.title(f"{model_name} — Top 15 Feature Importances (SHAP)",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, f"{model_name.lower()}_shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")

    # Beeswarm plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f"{model_name} — SHAP Beeswarm Plot",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(REPORTS_PATH, f"{model_name.lower()}_shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")

    return shap_values


def plot_shap_waterfall(model, X_test, model_name, n_samples=3):
    logger.info(f"Generating waterfall plots for {n_samples} sample predictions...")
    explainer = shap.TreeExplainer(model)

    for i in range(n_samples):
        sample = X_test.iloc[[i]]
        shap_vals = explainer(sample)

        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_vals[0], show=False, max_display=12)
        plt.title(f"{model_name} — Prediction Explanation (Sample {i+1})",
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(REPORTS_PATH,
                           f"{model_name.lower()}_waterfall_sample{i+1}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {path}")


def select_best_model():
    logger.info("Loading model comparison metrics...")
    with open("data/reports/model_comparison.json") as f:
        comparison = json.load(f)

    best_model_name = None
    best_roc = 0
    for name, metrics in comparison.items():
        roc = metrics.get("roc_auc", 0)
        logger.info(f"  {name}: ROC-AUC = {roc}")
        if roc > best_roc:
            best_roc = roc
            best_model_name = name

    logger.info(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_roc})")

    model_paths = {
        "LogisticRegression": "data/models/logistic_regression.pkl",
        "XGBoost": "data/models/xgboost_model.pkl",
        "LightGBM": "data/models/lightgbm_model.pkl"
    }

    best_model = joblib.load(model_paths[best_model_name])
    joblib.dump(best_model, "data/models/best_model.pkl")
    logger.info(f"Best model saved to: data/models/best_model.pkl")

    with open("data/reports/best_model_info.json", "w") as f:
        json.dump({
            "model_name": best_model_name,
            "roc_auc": best_roc,
            "metrics": comparison[best_model_name]
        }, f, indent=4)

    return best_model, best_model_name


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = get_test_set(X, y)

    # Select best model
    best_model, best_model_name = select_best_model()

    # SHAP analysis on best model
    shap_values = plot_shap_summary(best_model, X_test, best_model_name)
    plot_shap_waterfall(best_model, X_test, best_model_name)

    logger.info("\n✅ Evaluation complete!")
    logger.info(f"SHAP plots saved to: {REPORTS_PATH}")
    logger.info("Files generated:")
    for f in sorted(os.listdir(REPORTS_PATH)):
        logger.info(f"  - {f}")