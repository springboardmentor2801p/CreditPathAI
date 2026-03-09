# ── sys.path fix MUST be first ────────────────────────────────────────────────
import sys, os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Warning suppression ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*force_all_finite.*")

# ── Standard library ─────────────────────────────────────────────────────────
import json
import logging

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

# ── Local modules ─────────────────────────────────────────────────────────────
from src.phase4_risk_scoring.risk_scorer import RiskScorer
from src.phase4_risk_scoring.recommender import LoanRecommender
from src.phase4_risk_scoring.explainer   import RiskExplainer

# ── Output paths ──────────────────────────────────────────────────────────────
MODELS_DIR      = _PROJECT_ROOT / "models"
PHASE4_DIR      = MODELS_DIR   / "phase4_outputs"
REPORT_DIR      = PHASE4_DIR   / "reports"
SCORE_DIR       = PHASE4_DIR   / "scores"
EXPL_DIR        = PHASE4_DIR   / "explanations"
EXPERIMENTS_DIR = MODELS_DIR   / "experiments"
MLFLOW_DIR      = EXPERIMENTS_DIR / "mlflow"
DATA_DIR        = _PROJECT_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)


# ── Logging setup ─────────────────────────────────────────────────────────────
def setup_logging():
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
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
                str(EXPERIMENTS_DIR / "phase4_scoring.log"),
                mode="w", encoding="utf-8",
            ),
        ],
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_validation_data():
    """Load Phase 3 val.csv and split into features / labels."""
    val_path = DATA_DIR / "val.csv"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_path}\n"
            "Run Phase 3 (main.py) first."
        )
    df = pd.read_csv(val_path)
    _TARGETS = {"is_default", "default", "Default", "TARGET",
                "target", "loan_status", "bad_flag"}
    target = next((c for c in df.columns if c in _TARGETS), df.columns[-1])
    return df.drop(columns=[target]), df[target].values.astype(int), target


def _chart_risk_distribution(scored_df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    order   = ["Low", "Medium", "High", "Very High"]
    counts  = scored_df["risk_tier"].value_counts()
    vals    = [counts.get(t, 0) for t in order]
    colours = ["#2e7d32", "#f9a825", "#e65100", "#b71c1c"]
    total   = len(scored_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = axes[0].bar(order, vals, color=colours, edgecolor="white")
    axes[0].set_title("Risk Tier Distribution -- Validation Set", fontweight="bold")
    axes[0].set_xlabel("Risk Tier"); axes[0].set_ylabel("Applications")
    axes[0].grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.003,
            f"{v:,}\n({v/total*100:.1f}%)", ha="center", va="bottom", fontsize=9,
        )

    # Pie chart
    nz = [(t, c, col) for t, c, col in zip(order, vals, colours) if c > 0]
    axes[1].pie(
        [c for _, c, _ in nz],
        labels=[t for t, _, _ in nz],
        colors=[col for _, _, col in nz],
        autopct="%1.1f%%", startangle=140, textprops={"fontsize": 10},
    )
    axes[1].set_title("Risk Tier Proportions", fontweight="bold")

    plt.suptitle("CreditPathAI Phase 4 -- Risk Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "risk_tier_distribution.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    logger.info(f"Risk distribution chart saved {path}")
    return path


def _chart_score_histograms(scored_df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(scored_df["risk_score"], bins=50, color="steelblue", edgecolor="white")
    axes[0].axvline(scored_df["risk_score"].mean(), color="red", linestyle="--",
                    label=f"Mean = {scored_df['risk_score'].mean():.0f}")
    axes[0].set_xlabel("Risk Score (0-1000)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Risk Score Distribution"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(scored_df["p_default"], bins=50, color="coral", edgecolor="white")
    axes[1].axvline(scored_df["p_default"].mean(), color="red", linestyle="--",
                    label=f"Mean = {scored_df['p_default'].mean():.3f}")
    axes[1].set_xlabel("P(Default)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Default Probability Distribution"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("CreditPathAI Phase 4 -- Score & Probability Distributions", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "score_distributions.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    logger.info(f"Score histogram saved {path}")
    return path


def _chart_calibration(scored_df: pd.DataFrame, y_true: np.ndarray,
                        out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    edges = np.linspace(0, 1, 11)
    mids, actual, predicted = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (scored_df["p_default"] >= lo) & (scored_df["p_default"] < hi)
        if mask.sum() == 0:
            continue
        mids.append((lo + hi) / 2)
        actual.append(float(y_true[mask.values].mean()))
        predicted.append(float(scored_df["p_default"][mask].mean()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.plot(predicted, actual, "o-", color="steelblue", lw=2, ms=7, label="Model")
    ax.set_xlabel("Mean Predicted P(Default)")
    ax.set_ylabel("Actual Default Rate")
    ax.set_title("Model Calibration -- Reliability Diagram")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_chart.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    logger.info(f"Calibration chart saved {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    setup_logging()

    logger.info("=" * 80)
    logger.info("PHASE 4 -- CreditPathAI: Risk Scoring & Recommendation Engine")
    logger.info("=" * 80)

    for d in [PHASE4_DIR, REPORT_DIR, SCORE_DIR, EXPL_DIR, EXPERIMENTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Load Phase 3 best model ───────────────────────────────────────
    logger.info("[STEP 1] Loading Phase 3 best model...")
    scorer = RiskScorer()

    # ── STEP 2: Load validation set & batch score ──────────────────────────────
    logger.info("[STEP 2] Batch scoring validation set...")
    X_val_df, y_val, target_col = _load_validation_data()
    scored_df             = scorer.score_batch(X_val_df)
    scored_df[target_col] = y_val

    scored_csv = str(SCORE_DIR / "validation_scores.csv")
    scored_df.to_csv(scored_csv, index=False)
    logger.info(f"Scores saved -> {scored_csv}")

    # ── STEP 3: Risk distribution ──────────────────────────────────────────────
    logger.info("[STEP 3] Analysing risk distribution...")
    dist = scorer.get_risk_distribution(scored_df)
    for tier in ["Low", "Medium", "High", "Very High"]:
        cnt = dist["tier_counts"].get(tier, 0)
        pct = dist["tier_pct"].get(tier, 0.0)
        logger.info(f"  {tier:<12}: {cnt:>6,} ({pct:>5.1f}%)")
    logger.info(f"  Mean P(Default) : {dist['mean_p_default']:.4f}")
    logger.info(f"  Mean Risk Score : {dist['mean_risk_score']:.1f} / 1000")
    logger.info(f"  High+ Risk      : {dist['high_risk_pct']:.2f}%")

    # ── STEP 4: Visualisations ─────────────────────────────────────────────────
    logger.info("[STEP 4] Generating visualisations...")
    dist_path = _chart_risk_distribution(scored_df, str(REPORT_DIR))
    hist_path = _chart_score_histograms(scored_df, str(REPORT_DIR))
    cal_path  = _chart_calibration(scored_df, y_val, str(REPORT_DIR))

    # ── STEP 5: Sample recommendations (one per tier) ─────────────────────────
    logger.info("[STEP 5] Generating sample recommendations...")
    recommender  = LoanRecommender()
    sample_recs  = []

    for tier in ["Low", "Medium", "High", "Very High"]:
        mask = scored_df["risk_tier"] == tier
        if mask.sum() == 0:
            continue
        row     = scored_df[mask].iloc[0]
        profile = X_val_df[mask].iloc[0].to_dict()
        sr      = {
            "p_default":  float(row["p_default"]),
            "risk_score": int(row["risk_score"]),
            "risk_tier":  tier,
            "confidence": round(max(float(row["p_default"]),
                                    1.0 - float(row["p_default"])), 4),
        }
        rec = recommender.recommend(sr, profile)
        sample_recs.append(rec.to_dict())
        logger.info(
            f"  [{tier:<9}] {rec.decision:<22} "
            f"Rate: {rec.interest_rate_rec}%  Score: {rec.risk_score}"
        )

    recs_json = str(REPORT_DIR / "sample_recommendations.json")
    with open(recs_json, "w", encoding="utf-8") as fh:
        json.dump(sample_recs, fh, indent=2, ensure_ascii=False)
    logger.info(f"Sample recommendations saved -> {recs_json}")

    # ── STEP 6: SHAP explanations ──────────────────────────────────────────────
    logger.info("[STEP 6] Generating SHAP / importance explanations...")
    X_val_arr = scorer._align(X_val_df)

    explainer = RiskExplainer(
        model           = scorer.model,
        feature_names   = scorer.feature_names,
        background_data = X_val_arr[:200],
    )

    expl_paths = []
    for tier in ["Low", "Medium", "High", "Very High"]:
        mask = (scored_df["risk_tier"] == tier).values
        if mask.sum() == 0:
            continue
        row_idx  = int(np.where(mask)[0][0])
        expl     = explainer.explain_single(
            X_val_arr[row_idx],
            output_dir   = str(EXPL_DIR),
            applicant_id = f"sample_{tier.replace(' ', '_')}",
        )
        if expl.get("plot_path"):
            expl_paths.append(expl["plot_path"])
        top = expl["top_features"][0]["feature"] if expl["top_features"] else "N/A"
        logger.info(f"  [{tier:<9}] method={expl['method']}  top_factor={top}")

    # SHAP summary over first 500 rows
    summary_path = explainer.generate_summary_plot(
        X_val_arr[:500], output_dir=str(EXPL_DIR)
    )

    # ── STEP 7: Phase 4 report JSON ────────────────────────────────────────────
    logger.info("[STEP 7] Generating Phase 4 report...")
    report = {
        "phase":               "Phase 4 - Risk Scoring & Recommendation Engine",
        "model_type":          type(scorer.model).__name__,
        "features_used":       len(scorer.feature_names),
        "applications_scored": len(scored_df),
        "risk_distribution":   dist,
        "sample_recommendations": sample_recs,
        "artifacts": {
            "scored_csv":          scored_csv,
            "risk_dist_chart":     dist_path,
            "score_histogram":     hist_path,
            "calibration_chart":   cal_path,
            "recommendations_json": recs_json,
            "explanation_plots":   expl_paths,
            "shap_summary":        summary_path,
        },
    }
    report_json = str(REPORT_DIR / "phase4_report.json")
    with open(report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    logger.info(f"Report saved -> {report_json}")

    # ── STEP 8: MLflow logging ─────────────────────────────────────────────────
    logger.info("[STEP 8] Logging to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("CreditPathAI_Phase4")

    with mlflow.start_run(run_name="Phase4_RiskScoring"):
        mlflow.log_param("model_type",          type(scorer.model).__name__)
        mlflow.log_param("features_used",       len(scorer.feature_names))
        mlflow.log_param("applications_scored", len(scored_df))

        mlflow.log_metric("mean_p_default",  dist["mean_p_default"])
        mlflow.log_metric("mean_risk_score", dist["mean_risk_score"])
        mlflow.log_metric("high_risk_pct",   dist["high_risk_pct"])
        for tier, cnt in dist["tier_counts"].items():
            key = tier.lower().replace(" ", "_")
            mlflow.log_metric(f"tier_{key}_count", cnt)
            mlflow.log_metric(f"tier_{key}_pct",   dist["tier_pct"].get(tier, 0.0))

        all_artifacts = [
            (dist_path,  "visualizations"),
            (hist_path,  "visualizations"),
            (cal_path,   "visualizations"),
            (scored_csv, "scores"),
            (recs_json,  "reports"),
            (report_json,"reports"),
        ]
        for p in expl_paths:
            all_artifacts.append((p, "visualizations"))
        if summary_path:
            all_artifacts.append((summary_path, "visualizations"))

        for path, folder in all_artifacts:
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path=folder)

    logger.info("MLflow run logged successfully")

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("PHASE 4 COMPLETED SUCCESSFULLY!")
    logger.info("")
    logger.info("Deliverables:")
    logger.info(f"  Scored {len(scored_df):,} applications")
    logger.info(
        f"  Tiers: Low={dist['tier_counts'].get('Low',0):,} | "
        f"Medium={dist['tier_counts'].get('Medium',0):,} | "
        f"High={dist['tier_counts'].get('High',0):,} | "
        f"Very High={dist['tier_counts'].get('Very High',0):,}"
    )
    logger.info(f"  Risk charts          -> {REPORT_DIR}")
    logger.info(f"  Recommendations JSON -> {recs_json}")
    logger.info(f"  SHAP explanations    -> {EXPL_DIR}")
    logger.info(f"  Scored CSV           -> {scored_csv}")
    logger.info(f"  MLflow UI:")
    logger.info(f"    mlflow ui --backend-store-uri {MLFLOW_DIR.as_uri()}")
    logger.info("")
    logger.info(">>> Next Step: Phase 5 -- Dashboard & API Deployment")


if __name__ == "__main__":
    main()
