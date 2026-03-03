"""
================================================================================
  CreditPathAI — plot_utils.py
  Purpose : All chart generation + dark-theme styling helpers.
            No ML logic lives here — import and call from train_models.py.
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import ConfusionMatrixDisplay


# ─────────────────────────── Colour Palette ─────────────────────────────────
PALETTE = {
    "lr"      : "#4F46E5",   # indigo   – Logistic Regression
    "xgb"     : "#EC4899",   # pink     – XGBoost
    "lgb"     : "#10B981",   # emerald  – LightGBM
    "warn"    : "#F59E0B",   # amber
    "neutral" : "#6B7280",   # slate
    "bg"      : "#0F172A",   # dark navy background
    "surface" : "#1E293B",   # card / axes background
    "text"    : "#F1F5F9",   # near-white text
}

# Ordered list so callers can zip models → colours easily
MODEL_COLORS = [PALETTE["lr"], PALETTE["xgb"], PALETTE["lgb"]]


# ─────────────────────────── Theme Helpers ───────────────────────────────────
def apply_dark_theme(fig, axes):
    """Apply the CreditPathAI dark theme to a figure and one or many axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["text"])
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["neutral"])


def save_fig(fig, filename, output_dir):
    """Save figure to output_dir and close it."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  {path}")


def _legend_kw():
    """Shared legend style kwargs."""
    return dict(framealpha=0.3, labelcolor=PALETTE["text"],
                facecolor=PALETTE["surface"])


# ─────────────────────────── Individual Plot Functions ───────────────────────

def plot_class_distribution(y, output_dir):
    """Bar chart of target class imbalance."""
    from collections import Counter
    vc   = Counter(y)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Non-Default (0)", "Default (1)"],
        [vc[0], vc[1]],
        color=[PALETTE["lr"], PALETTE["xgb"]],
        width=0.5, edgecolor="none",
    )
    for bar, val in zip(bars, [vc[0], vc[1]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f"{val:,}", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=11)
    ax.set_title("Target Class Distribution (loanStatus)",
                 fontsize=13, pad=12)
    ax.set_ylabel("Count")
    apply_dark_theme(fig, ax)
    save_fig(fig, "01_class_distribution.png", output_dir)


def plot_roc_curves(results, output_dir):
    """
    Overlay ROC curves for all models.
    results : list of dicts from evaluate() in train_models.py
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for r, c in zip(results, MODEL_COLORS):
        ax.plot(r["fpr"], r["tpr"], color=c, lw=2.5,
                label=f"{r['name']}  (AUC = {r['auc_roc']:.4f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.2,
            ls="--", label="Random (0.50)")
    ax.fill_between(results[-1]["fpr"], results[-1]["tpr"],
                    alpha=0.06, color=MODEL_COLORS[-1])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", **_legend_kw())
    apply_dark_theme(fig, ax)
    save_fig(fig, "02_roc_comparison.png", output_dir)


def plot_pr_curves(results, baseline_prevalence, output_dir):
    """
    Overlay Precision-Recall curves for all models.
    baseline_prevalence : float  (positive-class rate in test set)
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for r, c in zip(results, MODEL_COLORS):
        ax.plot(r["rec"], r["prec"], color=c, lw=2.5,
                label=f"{r['name']}  (PR-AUC = {r['pr_auc']:.4f})")
    ax.axhline(baseline_prevalence, color=PALETTE["warn"], lw=1.4, ls="--",
               label=f"Baseline prevalence = {baseline_prevalence:.3f}")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", **_legend_kw())
    apply_dark_theme(fig, ax)
    save_fig(fig, "03_pr_comparison.png", output_dir)


def plot_confusion_matrices(results, output_dir):
    """Side-by-side confusion matrices for all models."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        disp = ConfusionMatrixDisplay(r["cm"],
                                      display_labels=["Non-Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(r["name"], fontsize=11)
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold",
                 color=PALETTE["text"], y=1.02)
    apply_dark_theme(fig, axes)
    save_fig(fig, "04_confusion_matrices.png", output_dir)


def plot_auc_bar(results, output_dir):
    """Grouped bar chart: AUC-ROC vs PR-AUC per model."""
    names  = [r["name"]    for r in results]
    aucs   = [r["auc_roc"] for r in results]
    praucs = [r["pr_auc"]  for r in results]
    x      = np.arange(len(names))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, aucs,  width=w, color=MODEL_COLORS[:len(results)],
                alpha=0.9, edgecolor="none", label="AUC-ROC")
    b2 = ax.bar(x + w/2, praucs, width=w, color=MODEL_COLORS[:len(results)],
                alpha=0.45, edgecolor="none", hatch="///", label="PR-AUC")
    for bar, v in zip(list(b1) + list(b2), aucs + praucs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f"{v:.4f}", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("AUC-ROC & PR-AUC Comparison", fontsize=14, fontweight="bold")
    ax.legend(**_legend_kw())
    apply_dark_theme(fig, ax)
    save_fig(fig, "05_auc_comparison.png", output_dir)


def plot_cv_folds(cv_results, output_dir):
    """
    Line chart: CV AUC-ROC per fold, one line per model.
    cv_results : dict  { model_name: sklearn cross_validate() output }
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, cv), c in zip(cv_results.items(), MODEL_COLORS):
        folds = cv["test_roc_auc"]
        ax.plot(range(len(folds)), folds,
                marker="o", color=c, lw=2, ms=6, label=label)
    n_folds = len(next(iter(cv_results.values()))["test_roc_auc"])
    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"{n_folds}-Fold CV AUC-ROC per Fold",
                 fontsize=14, fontweight="bold")
    ax.legend(**_legend_kw())
    apply_dark_theme(fig, ax)
    save_fig(fig, "06_cv_fold_comparison.png", output_dir)


def plot_score_distributions(results, y_test, output_dir):
    """
    Histogram of predicted P(Default) split by true class label.
    One subplot per model.
    """
    fig, axes = plt.subplots(1, len(results),
                              figsize=(6 * len(results), 5), sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, r, c in zip(axes, results, MODEL_COLORS):
        ax.hist(r["proba"][y_test == 0], bins=40, alpha=0.55,
                color=PALETTE["lr"], label="Non-Default", density=True)
        ax.hist(r["proba"][y_test == 1], bins=40, alpha=0.55,
                color=c,             label="Default",     density=True)
        ax.set_xlabel("Predicted P(Default)")
        ax.set_title(r["name"], fontsize=11)
        ax.legend(**_legend_kw())
    axes[0].set_ylabel("Density")
    fig.suptitle("Predicted Score Distribution by True Class",
                 fontsize=14, fontweight="bold", color=PALETTE["text"])
    apply_dark_theme(fig, axes)
    save_fig(fig, "07_score_distributions.png", output_dir)


def plot_feature_importance(feat_names, importances, model_name,
                             color, filename, output_dir, top_n=30):
    """
    Horizontal bar chart for a single model's feature importances.
    Works for both LR coefficients (signed) and tree gain/split (unsigned).
    """
    import pandas as pd
    df = pd.DataFrame({"feature": feat_names, "importance": importances})
    df["abs"] = df["importance"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11, 9))
    # Colour bars by sign if LR coefficients (can be negative)
    if (df["importance"] < 0).any():
        bar_colors = [PALETTE["xgb"] if v > 0 else PALETTE["lr"]
                      for v in df["importance"]]
        ax.barh(df["feature"], df["importance"],
                color=bar_colors, edgecolor="none")
        ax.axvline(0, color=PALETTE["neutral"], lw=1.0)
        pos_p = mpatches.Patch(color=PALETTE["xgb"], label="Increases default risk")
        neg_p = mpatches.Patch(color=PALETTE["lr"],  label="Reduces default risk")
        ax.legend(handles=[pos_p, neg_p], **_legend_kw())
        ax.set_xlabel("Coefficient Value", fontsize=12)
    else:
        ax.barh(df["feature"], df["importance"],
                color=color, edgecolor="none")
        ax.set_xlabel("Feature Importance", fontsize=12)

    ax.set_title(f"{model_name} — Top {top_n} Feature Importances",
                 fontsize=13)
    ax.invert_yaxis()
    apply_dark_theme(fig, ax)
    save_fig(fig, filename, output_dir)


def plot_grand_dashboard(results, cv_results, feat_names,
                          xgb_importances, lgb_importances,
                          y_test, output_dir):
    """
    2×3 summary dashboard combining all key charts in one figure.
    """
    import pandas as pd
    baseline_prev = y_test.mean()
    aucs   = [r["auc_roc"] for r in results]
    praucs = [r["pr_auc"]  for r in results]
    n_folds = len(next(iter(cv_results.values()))["test_roc_auc"])

    fig = plt.figure(figsize=(21, 14))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0, Col 0 — ROC
    ax00 = fig.add_subplot(gs[0, 0])
    for r, c in zip(results, MODEL_COLORS):
        ax00.plot(r["fpr"], r["tpr"], color=c, lw=2,
                  label=f"{r['name']} ({r['auc_roc']:.4f})")
    ax00.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1, ls="--")
    ax00.set_title("ROC Curve", fontsize=11)
    ax00.set_xlabel("FPR"); ax00.set_ylabel("TPR")
    ax00.legend(fontsize=7, **_legend_kw())

    # Row 0, Col 1 — PR
    ax01 = fig.add_subplot(gs[0, 1])
    for r, c in zip(results, MODEL_COLORS):
        ax01.plot(r["rec"], r["prec"], color=c, lw=2,
                  label=f"{r['name']} ({r['pr_auc']:.4f})")
    ax01.axhline(baseline_prev, color=PALETTE["warn"], lw=1, ls="--")
    ax01.set_title("Precision-Recall", fontsize=11)
    ax01.set_xlabel("Recall"); ax01.set_ylabel("Precision")
    ax01.legend(fontsize=7, **_legend_kw())

    # Row 0, Col 2 — AUC bar
    ax02 = fig.add_subplot(gs[0, 2])
    x    = np.arange(len(results))
    b1   = ax02.bar(x - 0.2, aucs,   0.35,
                    color=MODEL_COLORS[:len(results)], alpha=0.9,  edgecolor="none")
    b2   = ax02.bar(x + 0.2, praucs, 0.35,
                    color=MODEL_COLORS[:len(results)], alpha=0.4,
                    edgecolor="none", hatch="///")
    ax02.set_xticks(x)
    short = [r["name"].split()[0] for r in results]
    ax02.set_xticklabels(short, fontsize=9)
    ax02.set_ylim(0.5, 1.0)
    ax02.set_title("AUC Comparison", fontsize=11)
    ax02.set_ylabel("Score")
    for bar, v in zip(list(b1) + list(b2), aucs + praucs):
        ax02.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                  f"{v:.3f}", ha="center", va="bottom",
                  color=PALETTE["text"], fontsize=7)

    # Row 1, Col 0 — LightGBM importance (top 15)
    ax10 = fig.add_subplot(gs[1, 0])
    lgb_df = pd.DataFrame({"f": feat_names, "i": lgb_importances})
    lgb_df = lgb_df.nlargest(15, "i")
    ax10.barh(lgb_df["f"], lgb_df["i"], color=PALETTE["lgb"], edgecolor="none")
    ax10.set_title("LightGBM Top 15", fontsize=11)
    ax10.invert_yaxis()
    ax10.tick_params(axis="y", labelsize=7)

    # Row 1, Col 1 — XGBoost importance (top 15)
    ax11 = fig.add_subplot(gs[1, 1])
    xgb_df = pd.DataFrame({"f": feat_names, "i": xgb_importances})
    xgb_df = xgb_df.nlargest(15, "i")
    ax11.barh(xgb_df["f"], xgb_df["i"], color=PALETTE["xgb"], edgecolor="none")
    ax11.set_title("XGBoost Top 15", fontsize=11)
    ax11.invert_yaxis()
    ax11.tick_params(axis="y", labelsize=7)

    # Row 1, Col 2 — CV folds
    ax12 = fig.add_subplot(gs[1, 2])
    for (label, cv), c in zip(cv_results.items(), MODEL_COLORS):
        ax12.plot(range(n_folds), cv["test_roc_auc"],
                  marker="o", color=c, lw=2, ms=5, label=label)
    ax12.set_xticks(range(n_folds))
    ax12.set_xticklabels([f"F{i+1}" for i in range(n_folds)])
    ax12.set_title("CV AUC per Fold", fontsize=11)
    ax12.set_ylabel("AUC-ROC")
    ax12.legend(fontsize=7, **_legend_kw())

    all_axes = [ax00, ax01, ax02, ax10, ax11, ax12]
    apply_dark_theme(fig, all_axes)
    fig.suptitle("CreditPathAI — Model Comparison Dashboard",
                 fontsize=18, fontweight="bold",
                 color=PALETTE["text"], y=1.01)
    save_fig(fig, "08_grand_dashboard.png", output_dir)
