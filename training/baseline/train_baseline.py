"""
================================================================================
  CreditPathAI — baseline/train_baseline.py
  Model     : Logistic Regression  (class_weight='balanced')
  Tracking  : MLflow  experiment → "CreditPathAI_Baseline"
  Saves to  : baseline/saved_models/
  Plots to  : baseline/model_outputs/
================================================================================

USAGE
-----
    python baseline/train_baseline.py

MLflow UI
---------
    mlflow ui --backend-store-uri ./mlruns
    → open http://127.0.0.1:5000
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Allow imports from project root (feature_pipeline, plot_utils) ──────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score,
    ConfusionMatrixDisplay,
)

import feature_pipeline as fp
import plot_utils       as pu

# ─────────────────────────── Paths ───────────────────────────────────────────
THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(THIS_DIR, "saved_models")
OUTPUT_DIR  = os.path.join(THIS_DIR, "model_outputs")
MLRUNS_DIR  = os.path.join(ROOT, "mlruns")
DB_PATH     = os.path.join(ROOT, "creditpathai.db")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── Config ──────────────────────────────────────────
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
CV_FOLDS         = 5
EXPERIMENT_NAME  = "CreditPathAI_Baseline"


def section(t):
    print(f"\n{'═'*70}\n  {t}\n{'═'*70}")


# ═════════════════════════════════════════════════════════════════════════════
section("1 · Load & Engineer Features")
# ═════════════════════════════════════════════════════════════════════════════
X, y, preprocessor = fp.prepare(DB_PATH)

vc = y.value_counts()
print(f"\n  Non-default (0) : {vc[0]:,}  ({vc[0]/len(y)*100:.1f}%)")
print(f"  Default     (1) : {vc[1]:,}  ({vc[1]/len(y)*100:.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
section("2 · Train / Test Split")
# ═════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# ═════════════════════════════════════════════════════════════════════════════
section("3 · Preprocess")
# ═════════════════════════════════════════════════════════════════════════════
preprocessor.fit(X_train)
X_train_t  = preprocessor.transform(X_train)
X_test_t   = preprocessor.transform(X_test)
feat_names = preprocessor.get_feature_names_out()
print(f"  Shape after transform : {X_train_t.shape}")

# ═════════════════════════════════════════════════════════════════════════════
section("4 · Train Logistic Regression  [class_weight='balanced']")
# ═════════════════════════════════════════════════════════════════════════════

# ── Model hyperparameters ────────────────────────────────────────────────────
params = dict(
    penalty      = "l2",
    C            = 1.0,
    solver       = "lbfgs",
    max_iter     = 1000,
    class_weight = "balanced",
    random_state = RANDOM_STATE,
)

model = LogisticRegression(**params, n_jobs=-1)
model.fit(X_train_t, y_train)

# ── Metrics ──────────────────────────────────────────────────────────────────
proba    = model.predict_proba(X_test_t)[:, 1]
pred     = model.predict(X_test_t)
auc_roc  = roc_auc_score(y_test, proba)
pr_auc   = average_precision_score(y_test, proba)
f1       = f1_score(y_test, pred, zero_division=0)
cm       = confusion_matrix(y_test, pred)
fpr, tpr, _ = roc_curve(y_test, proba)
prec, rec, _ = precision_recall_curve(y_test, proba)

print(f"  AUC-ROC : {auc_roc:.4f}  |  PR-AUC : {pr_auc:.4f}  |  F1 : {f1:.4f}")
print(classification_report(y_test, pred,
      target_names=["Non-Default", "Default"], zero_division=0))

# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
skf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_res   = cross_validate(model, X_train_t, y_train,
                          cv=skf, n_jobs=-1,
                          scoring={"roc_auc": "roc_auc",
                                   "average_precision": "average_precision"})
cv_auc   = cv_res["test_roc_auc"]
cv_prauc = cv_res["test_average_precision"]
print(f"  CV AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV PR-AUC   : {cv_prauc.mean():.4f} ± {cv_prauc.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
section("5 · MLflow Tracking  →  CreditPathAI_Baseline")
# ═════════════════════════════════════════════════════════════════════════════
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="LogisticRegression_balanced"):

    # ── Log hyperparameters ──────────────────────────────────────────────────
    mlflow.log_params({
        "model"          : "LogisticRegression",
        "penalty"        : params["penalty"],
        "C"              : params["C"],
        "solver"         : params["solver"],
        "max_iter"       : params["max_iter"],
        "class_weight"   : params["class_weight"],
        "imbalance_tactic": "class_weight='balanced'",
        "test_size"      : TEST_SIZE,
        "cv_folds"       : CV_FOLDS,
        "n_features"     : len(feat_names),
        "train_rows"     : len(X_train),
        "test_rows"      : len(X_test),
    })

    # ── Log metrics ──────────────────────────────────────────────────────────
    mlflow.log_metrics({
        "auc_roc"      : auc_roc,
        "pr_auc"       : pr_auc,
        "f1_default"   : f1,
        "cv_auc_mean"  : cv_auc.mean(),
        "cv_auc_std"   : cv_auc.std(),
        "cv_prauc_mean": cv_prauc.mean(),
        "cv_prauc_std" : cv_prauc.std(),
        # Confusion matrix cells
        "tn" : int(cm[0, 0]), "fp" : int(cm[0, 1]),
        "fn" : int(cm[1, 0]), "tp" : int(cm[1, 1]),
    })

    # ── Log model ────────────────────────────────────────────────────────────
    mlflow.sklearn.log_model(model, artifact_path="model",
                             registered_model_name="LR_Baseline")
    print("  ✓ MLflow run logged (params + metrics + model).")

    # ── Log confusion matrix as artifact image ───────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Non-Default", "Default"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Logistic Regression", fontsize=11)
    cm_path = os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # ── Log ROC curve as artifact ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    pu.apply_dark_theme(fig, ax)
    ax.plot(fpr, tpr, color=pu.PALETTE["lr"], lw=2.5,
            label=f"LR Baseline  (AUC = {auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], color=pu.PALETTE["neutral"], lw=1.2, ls="--")
    ax.fill_between(fpr, tpr, alpha=0.10, color=pu.PALETTE["lr"])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curve — Logistic Regression Baseline", fontsize=12)
    ax.legend(loc="lower right", framealpha=0.3,
              labelcolor=pu.PALETTE["text"], facecolor=pu.PALETTE["surface"])
    roc_path = os.path.join(OUTPUT_DIR, "baseline_roc_curve.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    mlflow.log_artifact(roc_path, artifact_path="plots")

    # ── Log feature importance as artifact ────────────────────────────────────
    pu.plot_feature_importance(
        feat_names, model.coef_[0],
        model_name="Logistic Regression (Baseline)",
        color=pu.PALETTE["lr"],
        filename="baseline_feature_importance.png",
        output_dir=OUTPUT_DIR,
    )
    mlflow.log_artifact(
        os.path.join(OUTPUT_DIR, "baseline_feature_importance.png"),
        artifact_path="plots")

    run_id = mlflow.active_run().info.run_id
    print(f"  ✓ MLflow run ID : {run_id}")

# ═════════════════════════════════════════════════════════════════════════════
section("6 · Save Model to baseline/saved_models/")
# ═════════════════════════════════════════════════════════════════════════════

# Preprocessor
pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, pre_path)
print(f"  ✓  preprocessor.joblib           → {pre_path}")

# Model
mdl_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
joblib.dump(model, mdl_path)
print(f"  ✓  logistic_regression.joblib    → {mdl_path}")

# Metadata JSON
metadata = {
    "trained_at"       : datetime.now().isoformat(timespec="seconds"),
    "model_type"       : "baseline",
    "algorithm"        : "LogisticRegression",
    "imbalance_tactic" : "class_weight='balanced'",
    "n_features"       : int(len(feat_names)),
    "feature_names"    : list(feat_names),
    "train_rows"       : int(len(X_train)),
    "test_rows"        : int(len(X_test)),
    "random_state"     : RANDOM_STATE,
    "preprocessor_file": "preprocessor.joblib",
    "model_file"       : "logistic_regression.joblib",
    "mlflow_run_id"    : run_id,
    "mlflow_experiment": EXPERIMENT_NAME,
    "metrics": {
        "auc_roc"      : round(auc_roc, 4),
        "pr_auc"       : round(pr_auc, 4),
        "f1_default"   : round(f1, 4),
        "cv_auc_mean"  : round(cv_auc.mean(), 4),
        "cv_auc_std"   : round(cv_auc.std(), 4),
        "confusion_matrix": cm.tolist(),
    },
    "hyperparameters"  : params,
}

meta_path = os.path.join(MODELS_DIR, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓  metadata.json                 → {meta_path}")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  Baseline Model  (Logistic Regression)                      │
  │─────────────────────────────────────────────────────────────│
  │  AUC-ROC  : {auc_roc:.4f}   PR-AUC : {pr_auc:.4f}   F1 : {f1:.4f}       │
  │  CV AUC   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}                          │
  │─────────────────────────────────────────────────────────────│
  │  Saved to :  baseline/saved_models/                         │
  │  MLflow   :  mlruns/  (experiment: {EXPERIMENT_NAME})  │
  │─────────────────────────────────────────────────────────────│
  │  Next     :  python advanced/train_advanced.py              │
  └─────────────────────────────────────────────────────────────┘
""")
