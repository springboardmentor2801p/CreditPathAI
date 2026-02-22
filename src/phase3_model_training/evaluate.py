# ── Warning suppression ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Saving into deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

logger = logging.getLogger(__name__)


# ── Threshold optimisation ────────────────────────────────────────────────────
def find_best_f1_threshold(y_true, y_proba):
    """
    Sweep 0.10 – 0.90 in steps of 0.01 and return the threshold that
    maximises F1 on the minority (Default) class.
    For 10 % imbalance, optimal threshold typically falls at 0.25 – 0.35.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        score = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return round(best_t, 2), round(best_f1, 4)


# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate_model(model, X_val, y_val, model_name="Model",
                   threshold=None, output_dir="models/matrix"):
    """
    Full evaluation for one model.
    Saves: confusion_matrix (count + normalised) and threshold-analysis PNG.
    Returns a result dict (includes y_proba for downstream plotting).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Evaluating {model_name}")

    y_proba = model.predict_proba(X_val)[:, 1]

    if threshold is None:
        threshold, _ = find_best_f1_threshold(y_val, y_proba)
        logger.info(f"  Auto-selected threshold: {threshold}")

    y_pred = (y_proba >= threshold).astype(int)

    auc_roc       = float(roc_auc_score(y_val, y_proba))
    avg_precision = float(average_precision_score(y_val, y_proba))
    precision     = float(precision_score(y_val, y_pred, zero_division=0))
    recall        = float(recall_score(y_val, y_pred, zero_division=0))
    f1            = float(f1_score(y_val, y_pred, zero_division=0))
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info(f"AUC-ROC:       {auc_roc:.4f}")
    logger.info(f"Avg Precision: {avg_precision:.4f}")
    logger.info(f"Precision:     {precision:.4f}")
    logger.info(f"Recall:        {recall:.4f}")
    logger.info(f"F1 Score:      {f1:.4f}")
    logger.info(f"Specificity:   {specificity:.4f}")
    logger.info(f"Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    logger.info("\n" + classification_report(
        y_val, y_pred, target_names=["Current", "Default"]
    ))

    safe = model_name.replace(" ", "")
    tick = ["Current", "Default"]

    # ── Confusion matrix (count + normalised) ─────────────────────────────────
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=tick, yticklabels=tick, linewidths=0.5)
    axes[0].set_title(f"{model_name}\nConfusion Matrix — Counts  (t={threshold})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1],
                xticklabels=tick, yticklabels=tick, linewidths=0.5)
    axes[1].set_title(f"{model_name}\nConfusion Matrix — Normalised")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix_{safe}.png")
    plt.savefig(cm_path, dpi=130, bbox_inches="tight"); plt.close()
    logger.info(f"Confusion matrix saved {cm_path}")

    # ── Threshold analysis ────────────────────────────────────────────────────
    thr_range = np.arange(0.10, 0.91, 0.01)
    f1c, prec_c, rec_c = [], [], []
    for t in thr_range:
        p = (y_proba >= t).astype(int)
        f1c.append(f1_score(y_val, p, zero_division=0))
        prec_c.append(precision_score(y_val, p, zero_division=0))
        rec_c.append(recall_score(y_val, p, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thr_range, f1c,    label="F1 Score",  linewidth=2.5)
    ax.plot(thr_range, prec_c, label="Precision", linewidth=2, linestyle="--")
    ax.plot(thr_range, rec_c,  label="Recall",    linewidth=2, linestyle=":")
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"Best threshold = {threshold}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"{model_name} — Threshold Analysis")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    thr_path = os.path.join(output_dir, f"threshold_analysis_{safe}.png")
    plt.savefig(thr_path, dpi=130, bbox_inches="tight"); plt.close()

    return {
        "model_name":     model_name,
        "auc_roc":        round(auc_roc,       4),
        "avg_precision":  round(avg_precision, 4),
        "precision":      round(precision,     4),
        "recall":         round(recall,        4),
        "f1_score":       round(f1,            4),
        "specificity":    round(specificity,   4),
        "threshold":      threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "cm_path":        cm_path,
        "threshold_path": thr_path,
        "y_proba":        y_proba,
        "y_pred":         y_pred,
    }


# ── ROC + PR curves (all models, side by side) ────────────────────────────────
def generate_roc_pr_curves(models_results: dict, y_val,
                            output_dir="models/curves"):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, res in models_results.items():
        yp = res["y_proba"]
        fpr, tpr, _ = roc_curve(y_val, yp)
        prec, rec, _ = precision_recall_curve(y_val, yp)
        axes[0].plot(fpr, tpr, lw=2,
                     label=f"{name}  AUC={res['auc_roc']:.4f}")
        axes[1].plot(rec, prec, lw=2,
                     label=f"{name}  AP={res['avg_precision']:.4f}")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves — All Models")
    axes[0].legend(loc="lower right", fontsize=9); axes[0].grid(alpha=0.3)

    baseline = float(np.mean(y_val))
    axes[1].axhline(baseline, color="k", linestyle="--", lw=1,
                    label=f"No-skill ({baseline:.2%})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves — All Models")
    axes[1].legend(loc="upper right", fontsize=9); axes[1].grid(alpha=0.3)

    plt.suptitle("Model Comparison — ROC & PR Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_pr_curves_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    logger.info(f"ROC/PR curves saved {path}")
    return path


# ── Feature importance plots ─────────────────────────────────────────────────
def generate_feature_importance_plots(models: dict, feature_names: list,
                                       output_dir="models/matrix"):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, model in models.items():
        safe  = name.replace(" ", "")
        imps  = None
        title = "Feature Importances"

        if hasattr(model, "feature_importances_"):
            imps = model.feature_importances_
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            if clf is not None and hasattr(clf, "coef_"):
                imps  = np.abs(clf.coef_[0])
                title = "Feature Coefficients |coef|"

        if imps is None:
            logger.warning(f"No importances for {name}, skipping.")
            continue

        top_n   = min(20, len(imps))
        idx     = np.argsort(imps)[::-1][:top_n]
        top_f   = [feature_names[i] for i in idx]
        top_v   = imps[idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_n), top_v[::-1], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_f[::-1], fontsize=9)
        ax.set_xlabel("Score")
        ax.set_title(f"{name} — {title} (Top {top_n})")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        p = os.path.join(output_dir, f"feature_importance_{safe}.png")
        plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
        paths.append(p)
        logger.info(f"Feature importance saved {p}")
    return paths


# ── CV fold bar charts ────────────────────────────────────────────────────────
def generate_cv_fold_plots(cv_results: dict, output_dir="models/matrix"):
    """cv_results: {model_name: [fold_auc_1, fold_auc_2, ...]}"""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, fold_aucs in cv_results.items():
        safe     = name.replace(" ", "")
        folds    = [f"Fold {i+1}" for i in range(len(fold_aucs))]
        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(folds, fold_aucs, color="steelblue", edgecolor="white")
        ax.axhline(mean_auc, color="red", lw=1.5, linestyle="--",
                   label=f"Mean = {mean_auc:.4f} ± {std_auc:.4f}")
        y_lo = max(0.0, min(fold_aucs) - 0.02)
        y_hi = min(1.0, max(fold_aucs) + 0.02)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(f"{name} — K-Fold CV AUC  (k={len(fold_aucs)})")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, fold_aucs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (y_hi - y_lo) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        p = os.path.join(output_dir, f"cv_folds_{safe}.png")
        plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
        paths.append(p)
    return paths


# ── Model comparison bar chart ────────────────────────────────────────────────
def generate_model_comparison_bar(results_list: list,
                                   output_dir="models/curves"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["auc_roc", "f1_score", "precision", "recall", "avg_precision"]
    labels  = ["AUC-ROC", "F1", "Precision", "Recall", "Avg Prec"]
    names   = [r["model_name"] for r in results_list]
    x       = np.arange(len(labels))
    width   = 0.8 / len(names)
    offset  = -(len(names) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (res, mname) in enumerate(zip(results_list, names)):
        vals = [res[k] for k in metrics]
        bars = ax.bar(x + offset + i * width, vals, width,
                      label=mname, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylabel("Score"); ax.set_title("Model Comparison — Key Metrics")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.18); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(output_dir, "model_comparison_bar.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    logger.info(f"Model comparison chart saved {p}")
    return p


# ── Evaluation report CSV ─────────────────────────────────────────────────────
def generate_evaluation_report(results_list: list, cv_results: dict,
                                output_dir="models/experiments"):
    """
    cv_results: {name: (fold_aucs, mean_auc, std_auc)}
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for res in results_list:
        name = res["model_name"]
        cv_data = cv_results.get(name, ([], 0.0, 0.0))
        _, mean_auc, std_auc = cv_data if len(cv_data) == 3 else ([], 0.0, 0.0)
        rows.append({
            "Model":         name,
            "AUC_ROC":       res["auc_roc"],
            "Avg_Precision": res["avg_precision"],
            "Precision":     res["precision"],
            "Recall":        res["recall"],
            "F1_Score":      res["f1_score"],
            "Specificity":   res["specificity"],
            "Threshold":     res["threshold"],
            "CV_AUC_Mean":   round(float(mean_auc), 4),
            "CV_AUC_Std":    round(float(std_auc),  4),
            "TP": res["tp"], "FP": res["fp"],
            "FN": res["fn"], "TN": res["tn"],
        })

    df = pd.DataFrame(rows).sort_values("F1_Score", ascending=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 70)
    logger.info(df[["Model", "AUC_ROC", "Precision", "Recall",
                     "F1_Score", "CV_AUC_Mean", "CV_AUC_Std"]].to_string(index=False))
    logger.info("=" * 70)

    csv_path = os.path.join(output_dir, "evaluation_report.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Report saved {csv_path}")
    return df, csv_path


# ── MLflow child-run logger ───────────────────────────────────────────────────
def log_model_to_mlflow(model, model_name: str, result: dict,
                         fold_aucs: list, artifact_paths: list):
    """
    Log one model as a nested MLflow run.
    Call from inside an active parent mlflow.start_run() context.
    Logs: all metrics, per-fold AUC (stepped), PNG artifacts, CSV report,
          and the native model object (xgboost/lightgbm/sklearn format).
    """
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    import xgboost as xgb_lib
    import lightgbm as lgb_lib

    run_name = model_name.replace(" ", "_")
    mean_auc = round(float(np.mean(fold_aucs)), 4) if fold_aucs else 0.0
    std_auc  = round(float(np.std(fold_aucs)),  4) if fold_aucs else 0.0

    with mlflow.start_run(run_name=run_name, nested=True):
        # Parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("threshold",  result["threshold"])

        # Metrics
        for key in ("auc_roc", "avg_precision", "precision",
                    "recall", "f1_score", "specificity"):
            mlflow.log_metric(key, result[key])
        mlflow.log_metric("cv_auc_mean", mean_auc)
        mlflow.log_metric("cv_auc_std",  std_auc)
        for key in ("tp", "fp", "fn", "tn"):
            mlflow.log_metric(key, result[key])

        # Per-fold AUC → renders as a stepped line chart in MLflow UI
        for step, auc in enumerate(fold_aucs, 1):
            mlflow.log_metric("cv_fold_auc", auc, step=step)

        # Artifact files
        for path in artifact_paths:
            if not path or not os.path.exists(path):
                continue
            if path.endswith(".png"):
                mlflow.log_artifact(path, artifact_path="visualizations")
            elif path.endswith(".csv"):
                mlflow.log_artifact(path, artifact_path="reports")
            else:
                mlflow.log_artifact(path)

        # Native model object
        try:
            if isinstance(model, xgb_lib.XGBClassifier):
                mlflow.xgboost.log_model(
                    model, artifact_path="model", model_format="json"
                )
            elif isinstance(model, lgb_lib.LGBMClassifier):
                mlflow.lightgbm.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as exc:
            logger.warning(f"Could not log model object for {model_name}: {exc}")

    logger.info(f"MLflow run logged {model_name}")
