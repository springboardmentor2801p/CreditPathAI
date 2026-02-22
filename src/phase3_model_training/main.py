# ════════════════════════════════════════════════════════════════════════════
# CRITICAL: sys.path fix MUST be the very first executable lines.
# Adds D:\Infosys_Intern\CreditPathAI to sys.path so that
# `from src.phase3_model_training.xxx import ...` works when the script is
# run as  python src\phase3_model_training\main.py  from the project root.
# ════════════════════════════════════════════════════════════════════════════
import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Warning suppression (before every other import) ──────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Saving into deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Scoring failed.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Distutils.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Setuptools.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*psutil.*")

# ── Standard library ─────────────────────────────────────────────────────────
import json
import logging
import pickle

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from imblearn.over_sampling import SMOTE

# ── Local modules (safe now that sys.path is fixed) ──────────────────────────
from src.phase3_model_training.data_loader import load_processed_data
from src.phase3_model_training.train import (
    train_logistic_regression,
    train_xgboost,
    train_lightgbm,
    cross_validate_model,
    EnsembleModel,
    save_artifacts,
)
from src.phase3_model_training.evaluate import (
    evaluate_model,
    generate_roc_pr_curves,
    generate_feature_importance_plots,
    generate_cv_fold_plots,
    generate_model_comparison_bar,
    generate_evaluation_report,
    log_model_to_mlflow,
)

# ── Output directory constants (all relative to project root) ────────────────
MODELS_DIR      = _PROJECT_ROOT / "models"
MATRIX_DIR      = MODELS_DIR / "matrix"
CURVES_DIR      = MODELS_DIR / "curves"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
MLFLOW_DIR      = EXPERIMENTS_DIR / "mlflow"

logger = logging.getLogger(__name__)


def setup_logging():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fix Windows CP1252 terminal: rewrap stdout as UTF-8
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(EXPERIMENTS_DIR / "phase3_training.log"),
                mode="w",
                encoding="utf-8",   # log file always UTF-8
            ),
        ],
    )


def _select_best_model(results: dict, cv_results: dict) -> tuple:
    """
    Composite score = 0.4 * val_AUC + 0.3 * F1 + 0.2 * CV_AUC + 0.1 * recall
    Returns (best_name, best_model_key, composite_scores_dict).
    """
    scores = {}
    logger.info("")
    logger.info("=" * 65)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 65)
    logger.info(f"{'Model':<22} {'Val AUC':>8} {'CV AUC':>8} "
                f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
    logger.info("-" * 65)

    for name, res in results.items():
        _, cv_mean, _ = cv_results.get(name, ([], 0.0, 0.0))
        composite = (
            0.40 * res["auc_roc"]
            + 0.30 * res["f1_score"]
            + 0.20 * float(cv_mean)
            + 0.10 * res["recall"]
        )
        scores[name] = composite
        logger.info(
            f"{name:<22} {res['auc_roc']:>8.4f} {float(cv_mean):>8.4f} "
            f"{res['precision']:>10.4f} {res['recall']:>8.4f} "
            f"{res['f1_score']:>8.4f}"
        )

    logger.info("=" * 65)
    best_name = max(scores, key=scores.__getitem__)
    logger.info(f"Best model: {best_name}  (composite score: {scores[best_name]:.4f})")
    return best_name, scores


def main():
    setup_logging()

    logger.info("=" * 80)
    logger.info("PHASE 3 — CreditPathAI Model Training  [IMPROVED]")
    logger.info("=" * 80)

    for d in [MODELS_DIR, MATRIX_DIR, CURVES_DIR, EXPERIMENTS_DIR, MLFLOW_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load data ──────────────────────────────────────────────────────
    logger.info("[STEP 1] Loading processed datasets...")
    X_train, X_val, y_train, y_val, feature_names = load_processed_data()
    dataset_size = len(X_train) + len(X_val)
    num_features = X_train.shape[1]
    default_rate = float(y_val.mean() * 100)
    logger.info(
        f"Total: {dataset_size} | Features: {num_features} "
        f"| Default rate: {default_rate:.2f}%"
    )

    # ── STEP 2: SMOTE ──────────────────────────────────────────────────────────
    logger.info("[STEP 2] Applying SMOTE to training set...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    logger.info(
        f"SMOTE: {len(X_train)} -> {len(X_sm)} training samples "
        f"(pos={int((y_sm==1).sum())}, neg={int((y_sm==0).sum())})"
    )

    # ── STEP 3: Logistic Regression ────────────────────────────────────────────
    logger.info("[STEP 3] Training Logistic Regression...")
    lr_model = train_logistic_regression(X_sm, y_sm)

    # ── STEP 4: XGBoost ────────────────────────────────────────────────────────
    logger.info("[STEP 4] Training XGBoost...")
    xgb_model = train_xgboost(X_sm, y_sm)

    # ── STEP 5: LightGBM ───────────────────────────────────────────────────────
    logger.info("[STEP 5] Training LightGBM...")
    lgb_model = train_lightgbm(X_sm, y_sm)

    base_models = {
        "Logistic Regression": lr_model,
        "XGBoost":             xgb_model,
        "LightGBM":            lgb_model,
    }
    ensemble    = EnsembleModel(base_models)
    all_models  = {**base_models, "Ensemble": ensemble}

    # ── STEP 6: K-Fold CV ──────────────────────────────────────────────────────
    logger.info(f"[STEP 6] K-Fold Cross-Validation (k=5)...")
    cv_results = {}
    for name, model in base_models.items():
        logger.info(f"  CV {name}")
        fold_aucs, mean_auc, std_auc = cross_validate_model(
            model, X_sm, y_sm, n_splits=5, model_name=name
        )
        cv_results[name] = (fold_aucs, mean_auc, std_auc)

    # Approximate ensemble CV as average of base fold AUCs
    ens_folds = [
        float(np.mean([cv_results[n][0][i] for n in base_models]))
        for i in range(5)
    ]
    cv_results["Ensemble"] = (
        ens_folds, float(np.mean(ens_folds)), float(np.std(ens_folds))
    )

    # ── STEP 7: Evaluate on validation set ────────────────────────────────────
    logger.info("[STEP 7] Evaluating on validation set...")
    eval_results = {}
    for name, model in all_models.items():
        res = evaluate_model(
            model=model, X_val=X_val, y_val=y_val,
            model_name=name, threshold=None,
            output_dir=str(MATRIX_DIR),
        )
        eval_results[name] = res

    # ── STEP 8: Generate evaluation report ────────────────────────────────────
    logger.info("[STEP 8] Generating evaluation report...")
    report_df, report_csv = generate_evaluation_report(
        list(eval_results.values()), cv_results,
        output_dir=str(EXPERIMENTS_DIR),
    )

    # ── STEP 9: Log to MLflow ─────────────────────────────────────────────────
    logger.info("[STEP 9] Logging experiments to MLflow...")
    dataset_params = {
        "dataset_size":     dataset_size,
        "num_features":     num_features,
        "default_rate_pct": round(default_rate, 2),
        "smote_applied":    True,
        "cv_folds":         5,
    }

    # Generate all visualisation artifacts once (shared across runs)
    roc_pr_path = generate_roc_pr_curves(
        eval_results, y_val, output_dir=str(CURVES_DIR)
    )
    fi_paths = generate_feature_importance_plots(
        all_models, feature_names, output_dir=str(MATRIX_DIR)
    )
    cv_fold_paths = generate_cv_fold_plots(
        {n: cv_results[n][0] for n in base_models},
        output_dir=str(MATRIX_DIR),
    )
    comp_path = generate_model_comparison_bar(
        list(eval_results.values()), output_dir=str(CURVES_DIR)
    )

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("CreditPathAI_Phase3")

    with mlflow.start_run(run_name="Phase3_Experiment"):
        for k, v in dataset_params.items():
            mlflow.log_param(k, v)

        for name, res in eval_results.items():
            model     = all_models[name]
            fold_aucs = cv_results[name][0]

            artifact_paths = []
            for key in ("cm_path", "threshold_path"):
                p = res.get(key, "")
                if p and os.path.exists(p):
                    artifact_paths.append(p)
            for p in fi_paths:
                if name.replace(" ", "") in os.path.basename(p):
                    artifact_paths.append(p)
            for p in cv_fold_paths:
                if name.replace(" ", "") in os.path.basename(p):
                    artifact_paths.append(p)
            artifact_paths += [roc_pr_path, comp_path, report_csv]

            log_model_to_mlflow(
                model=model, model_name=name, result=res,
                fold_aucs=fold_aucs, artifact_paths=artifact_paths,
            )

    # ── STEP 10: Select and save best model ────────────────────────────────────
    logger.info("[STEP 10] Selecting and saving best model...")
    best_name, composite_scores = _select_best_model(eval_results, cv_results)
    best_model = all_models[best_name]

    best_model_path = MODELS_DIR / "model.pkl"
    with open(best_model_path, "wb") as fh:
        pickle.dump(best_model, fh)
    logger.info(f"Best model saved {best_model_path}")

    metadata = {
        "best_model":        best_name,
        "composite_scores":  {k: round(v, 4) for k, v in composite_scores.items()},
        "metrics":           {
            name: {
                k: eval_results[name][k]
                for k in ("auc_roc", "f1_score", "precision",
                          "recall", "threshold", "avg_precision")
            }
            for name in eval_results
        },
        "dataset_params": dataset_params,
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"Metadata saved {meta_path}")

    # ── STEP 11: Save training artifacts ──────────────────────────────────────
    logger.info("[STEP 11] Saving training artifacts...")
    save_artifacts(all_models, output_dir=str(MODELS_DIR))

    logger.info("")
    logger.info("PHASE 3 COMPLETED SUCCESSFULLY!")
    logger.info("")
    logger.info("Deliverables:")
    logger.info(f"  3 base models + Ensemble trained")
    logger.info(f"  SMOTE applied — balanced training set")
    logger.info(f"  K-Fold CV (k=5) — all base models")
    logger.info(f"  Confusion matrices  -> {MATRIX_DIR}")
    logger.info(f"  ROC + PR curves     -> {CURVES_DIR}")
    logger.info(f"  Evaluation report   -> {EXPERIMENTS_DIR / 'evaluation_report.csv'}")
    logger.info(f"  Best model          -> {best_name}  "
                f"(F1={eval_results[best_name]['f1_score']:.4f})")
    logger.info(f"  Best model saved    -> {best_model_path}")
    logger.info(f"  MLflow experiments  -> {MLFLOW_DIR}")
    logger.info(f"  View MLflow UI:")
    logger.info(f"    mlflow ui --backend-store-uri {MLFLOW_DIR.as_uri()}")
    logger.info("")
    logger.info(">>> Next Step: Phase 4 — Risk Scoring & Recommendation Engine")


if __name__ == "__main__":
    main()
