"""
================================================================================
  CreditPathAI — advanced/train_advanced.py
  Models    : XGBoost  (scale_pos_weight=9)
              LightGBM (is_unbalance=True)
  Tracking  : MLflow  experiment → "CreditPathAI_Advanced"
  Saves to  : advanced/saved_models/
  Plots to  : advanced/model_outputs/
================================================================================

USAGE
-----
    python advanced/train_advanced.py

MLflow UI
---------
    mlflow ui --backend-store-uri ./mlruns
    → open http://127.0.0.1:5000
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics         import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score,
    ConfusionMatrixDisplay,
)
import xgboost  as xgb
import lightgbm as lgb

import feature_pipeline as fp
import plot_utils       as pu

# ─────────────────────────── Paths ───────────────────────────────────────────
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "saved_models")
OUTPUT_DIR = os.path.join(THIS_DIR, "model_outputs")
MLRUNS_DIR = os.path.join(ROOT, "mlruns")
DB_PATH    = os.path.join(ROOT, "creditpathai.db")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── Config ──────────────────────────────────────────
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
CV_FOLDS         = 5
IMBALANCE_RATIO  = 9
EXPERIMENT_NAME  = "CreditPathAI_Advanced"


def section(t):
    print(f"\n{'═'*70}\n  {t}\n{'═'*70}")


def evaluate(name, model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    pred  = model.predict(X_te)
    fpr, tpr, _  = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    return {
        "name"   : name,
        "auc_roc": roc_auc_score(y_te, proba),
        "pr_auc" : average_precision_score(y_te, proba),
        "f1"     : f1_score(y_te, pred, zero_division=0),
        "proba"  : proba, "pred": pred,
        "fpr"    : fpr,   "tpr" : tpr,
        "prec"   : prec,  "rec" : rec,
        "cm"     : confusion_matrix(y_te, pred),
        "report" : classification_report(y_te, pred,
                       target_names=["Non-Default", "Default"],
                       zero_division=0),
    }


def log_model_run(run_name, model_obj, model_label,
                  params, res, cv_auc, cv_prauc,
                  output_dir, color, log_fn):
    """
    Open an MLflow run and log: params, metrics, model, and plot artifacts.
    Returns the MLflow run_id string.
    """
    cm = res["cm"]
    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_params({**params,
                           "model": model_label,
                           "cv_folds": CV_FOLDS})

        # Metrics
        mlflow.log_metrics({
            "auc_roc"      : res["auc_roc"],
            "pr_auc"       : res["pr_auc"],
            "f1_default"   : res["f1"],
            "cv_auc_mean"  : cv_auc.mean(),
            "cv_auc_std"   : cv_auc.std(),
            "cv_prauc_mean": cv_prauc.mean(),
            "cv_prauc_std" : cv_prauc.std(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        })

        # Model artifact
        log_fn(model_obj, artifact_path="model")

        # ── Confusion Matrix ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["Non-Default", "Default"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {model_label}", fontsize=11)
        slug     = model_label.lower().replace(" ", "_")
        cm_path  = os.path.join(output_dir, f"{slug}_confusion_matrix.png")
        fig.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ── ROC Curve ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        pu.apply_dark_theme(fig, ax)
        ax.plot(res["fpr"], res["tpr"], color=color, lw=2.5,
                label=f"{model_label}  (AUC = {res['auc_roc']:.4f})")
        ax.plot([0, 1], [0, 1], color=pu.PALETTE["neutral"], lw=1.2, ls="--")
        ax.fill_between(res["fpr"], res["tpr"], alpha=0.10, color=color)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate",  fontsize=11)
        ax.set_title(f"ROC Curve — {model_label}", fontsize=12)
        ax.legend(loc="lower right", framealpha=0.3,
                  labelcolor=pu.PALETTE["text"], facecolor=pu.PALETTE["surface"])
        roc_path = os.path.join(output_dir, f"{slug}_roc_curve.png")
        fig.savefig(roc_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor()); plt.close(fig)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        run_id = mlflow.active_run().info.run_id

    return run_id


# ═════════════════════════════════════════════════════════════════════════════
section("1 · Load & Engineer Features")
X, y, preprocessor = fp.prepare(DB_PATH)
vc = y.value_counts()
print(f"\n  Non-default (0) : {vc[0]:,}  ({vc[0]/len(y)*100:.1f}%)")
print(f"  Default     (1) : {vc[1]:,}  ({vc[1]/len(y)*100:.1f}%)")

section("2 · Train / Test Split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

section("3 · Preprocess")
preprocessor.fit(X_train)
X_train_t  = preprocessor.transform(X_train)
X_test_t   = preprocessor.transform(X_test)
feat_names = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_t, columns=feat_names)   # named — avoids LGB warning
X_test_df  = pd.DataFrame(X_test_t,  columns=feat_names)
print(f"  Shape after transform : {X_train_t.shape}")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ═════════════════════════════════════════════════════════════════════════════
section("4 · XGBoost  [scale_pos_weight=9]")
# ═════════════════════════════════════════════════════════════════════════════
xgb_params = dict(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_child_weight = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    gamma            = 0.1,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    scale_pos_weight = IMBALANCE_RATIO,
    eval_metric      = "auc",
    tree_method      = "hist",
    random_state     = RANDOM_STATE,
)
xgb_model = xgb.XGBClassifier(**xgb_params, n_jobs=-1)
print("  Training …")
xgb_model.fit(X_train_t, y_train,
              eval_set=[(X_test_t, y_test)], verbose=False)

res_xgb    = evaluate("XGBoost", xgb_model, X_test_t, y_test)
cv_xgb     = cross_validate(xgb_model, X_train_df, y_train, cv=skf, n_jobs=-1,
                             scoring={"roc_auc": "roc_auc",
                                      "average_precision": "average_precision"})
cv_xgb_auc = cv_xgb["test_roc_auc"]
cv_xgb_pr  = cv_xgb["test_average_precision"]

print(f"  AUC-ROC : {res_xgb['auc_roc']:.4f}  |  PR-AUC : {res_xgb['pr_auc']:.4f}  "
      f"|  F1 : {res_xgb['f1']:.4f}")
print(f"  CV AUC  : {cv_xgb_auc.mean():.4f} ± {cv_xgb_auc.std():.4f}")
print(res_xgb["report"])

# ── Feature importance plot ──────────────────────────────────────────────────
pu.plot_feature_importance(
    feat_names, xgb_model.feature_importances_,
    model_name="XGBoost", color=pu.PALETTE["xgb"],
    filename="xgboost_feature_importance.png", output_dir=OUTPUT_DIR)


# ═════════════════════════════════════════════════════════════════════════════
section("5 · LightGBM  [is_unbalance=True]")
# ═════════════════════════════════════════════════════════════════════════════
lgb_params = dict(
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    is_unbalance      = True,
    metric            = "auc",
    random_state      = RANDOM_STATE,
    verbose           = -1,
)
lgb_model = lgb.LGBMClassifier(**lgb_params, n_jobs=-1)
print("  Training …")
lgb_model.fit(X_train_df, y_train,
              eval_set=[(X_test_df, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

res_lgb    = evaluate("LightGBM", lgb_model, X_test_df, y_test)
cv_lgb     = cross_validate(lgb_model, X_train_df, y_train, cv=skf, n_jobs=-1,
                             scoring={"roc_auc": "roc_auc",
                                      "average_precision": "average_precision"})
cv_lgb_auc = cv_lgb["test_roc_auc"]
cv_lgb_pr  = cv_lgb["test_average_precision"]

print(f"  AUC-ROC : {res_lgb['auc_roc']:.4f}  |  PR-AUC : {res_lgb['pr_auc']:.4f}  "
      f"|  F1 : {res_lgb['f1']:.4f}")
print(f"  CV AUC  : {cv_lgb_auc.mean():.4f} ± {cv_lgb_auc.std():.4f}")
print(res_lgb["report"])

pu.plot_feature_importance(
    feat_names, lgb_model.feature_importances_,
    model_name="LightGBM", color=pu.PALETTE["lgb"],
    filename="lightgbm_feature_importance.png", output_dir=OUTPUT_DIR)


# ═════════════════════════════════════════════════════════════════════════════
section("6 · Comparison Plot  (XGBoost vs LightGBM)")
# ═════════════════════════════════════════════════════════════════════════════
results = [res_xgb, res_lgb]

pu.plot_roc_curves(results, OUTPUT_DIR)
pu.plot_pr_curves(results, baseline_prevalence=y_test.mean(),
                  output_dir=OUTPUT_DIR)
pu.plot_confusion_matrices(results, OUTPUT_DIR)
pu.plot_auc_bar(results, OUTPUT_DIR)


# ═════════════════════════════════════════════════════════════════════════════
section("7 · MLflow Tracking  →  CreditPathAI_Advanced")
# ═════════════════════════════════════════════════════════════════════════════
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")
mlflow.set_experiment(EXPERIMENT_NAME)

xgb_run_id = log_model_run(
    run_name    = "XGBoost_scale_pos_weight9",
    model_obj   = xgb_model,
    model_label = "XGBoost",
    params      = {**xgb_params, "imbalance_tactic": f"scale_pos_weight={IMBALANCE_RATIO}"},
    res         = res_xgb,
    cv_auc      = cv_xgb_auc,
    cv_prauc    = cv_xgb_pr,
    output_dir  = OUTPUT_DIR,
    color       = pu.PALETTE["xgb"],
    log_fn      = mlflow.xgboost.log_model,
)
print(f"  ✓ XGBoost  MLflow run ID : {xgb_run_id}")

lgb_run_id = log_model_run(
    run_name    = "LightGBM_is_unbalance",
    model_obj   = lgb_model,
    model_label = "LightGBM",
    params      = {**lgb_params, "imbalance_tactic": "is_unbalance=True"},
    res         = res_lgb,
    cv_auc      = cv_lgb_auc,
    cv_prauc    = cv_lgb_pr,
    output_dir  = OUTPUT_DIR,
    color       = pu.PALETTE["lgb"],
    log_fn      = mlflow.lightgbm.log_model,
)
print(f"  ✓ LightGBM MLflow run ID : {lgb_run_id}")


# ═════════════════════════════════════════════════════════════════════════════
section("8 · Save to advanced/saved_models/")
# ═════════════════════════════════════════════════════════════════════════════

# Shared preprocessor
pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, pre_path)
print(f"  ✓  preprocessor.joblib  → {pre_path}")

# XGBoost
xgb_path = os.path.join(MODELS_DIR, "xgboost.joblib")
joblib.dump(xgb_model, xgb_path)
print(f"  ✓  xgboost.joblib       → {xgb_path}")

# LightGBM
lgb_path = os.path.join(MODELS_DIR, "lightgbm.joblib")
joblib.dump(lgb_model, lgb_path)
print(f"  ✓  lightgbm.joblib      → {lgb_path}")

# Determine winner by AUC-ROC
winner_key  = "xgboost" if res_xgb["auc_roc"] >= res_lgb["auc_roc"] else "lightgbm"
winner_file = f"{winner_key}.joblib"

# Metadata
metadata = {
    "trained_at"        : datetime.now().isoformat(timespec="seconds"),
    "model_type"        : "advanced",
    "random_state"      : RANDOM_STATE,
    "n_features"        : int(len(feat_names)),
    "feature_names"     : list(feat_names),
    "train_rows"        : int(len(X_train)),
    "test_rows"         : int(len(X_test)),
    "preprocessor_file" : "preprocessor.joblib",
    "best_model"        : winner_key,
    "models": {
        "xgboost": {
            "filename"      : "xgboost.joblib",
            "mlflow_run_id" : xgb_run_id,
            "imbalance_tactic": f"scale_pos_weight={IMBALANCE_RATIO}",
            "hyperparameters": xgb_params,
            "metrics": {
                "auc_roc"      : round(res_xgb["auc_roc"], 4),
                "pr_auc"       : round(res_xgb["pr_auc"],  4),
                "f1_default"   : round(res_xgb["f1"],      4),
                "cv_auc_mean"  : round(cv_xgb_auc.mean(),  4),
                "cv_auc_std"   : round(cv_xgb_auc.std(),   4),
                "confusion_matrix": res_xgb["cm"].tolist(),
            },
        },
        "lightgbm": {
            "filename"      : "lightgbm.joblib",
            "mlflow_run_id" : lgb_run_id,
            "imbalance_tactic": "is_unbalance=True",
            "hyperparameters": lgb_params,
            "metrics": {
                "auc_roc"      : round(res_lgb["auc_roc"], 4),
                "pr_auc"       : round(res_lgb["pr_auc"],  4),
                "f1_default"   : round(res_lgb["f1"],      4),
                "cv_auc_mean"  : round(cv_lgb_auc.mean(),  4),
                "cv_auc_std"   : round(cv_lgb_auc.std(),   4),
                "confusion_matrix": res_lgb["cm"].tolist(),
            },
        },
    },
    "mlflow_experiment" : EXPERIMENT_NAME,
}

meta_path = os.path.join(MODELS_DIR, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓  metadata.json        → {meta_path}")

# ─────────────────────────── Final Summary ───────────────────────────────────
print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  Advanced Models — Results                                      │
  │─────────────────────────────────────────────────────────────────│
  │  {'Model':<16} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1(def)':>8} {'CV AUC':>8}     │
  │  {'XGBoost':<16} {res_xgb['auc_roc']:>8.4f} {res_xgb['pr_auc']:>8.4f} {res_xgb['f1']:>8.4f} {cv_xgb_auc.mean():>8.4f}     │
  │  {'LightGBM':<16} {res_lgb['auc_roc']:>8.4f} {res_lgb['pr_auc']:>8.4f} {res_lgb['f1']:>8.4f} {cv_lgb_auc.mean():>8.4f}     │
  │─────────────────────────────────────────────────────────────────│
  │  Winner   : {winner_key:<51}│
  │  Saved to : advanced/saved_models/                              │
  │  MLflow   : mlruns/  (experiment: {EXPERIMENT_NAME})  │
  │─────────────────────────────────────────────────────────────────│
  │  Predict  : python predict.py --tier advanced                   │
  │  MLflow UI: mlflow ui --backend-store-uri ./mlruns              │
  └─────────────────────────────────────────────────────────────────┘
""")
