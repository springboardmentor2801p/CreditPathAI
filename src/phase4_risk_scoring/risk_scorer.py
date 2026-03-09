# ── Warning suppression ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ─────────────────────────────────────────────────────────────────────────────

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FILE_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent.parent

# ── Risk tier boundaries on P(default) ───────────────────────────────────────
TIER_THRESHOLDS = [
    ("Low",       0.00, 0.15),
    ("Medium",    0.15, 0.35),
    ("High",      0.35, 0.60),
    ("Very High", 0.60, 1.01),
]


def _prob_to_score(p: float) -> int:
    """Convert P(default) to 0-1000 risk score (1000 = safest)."""
    return int(round((1.0 - float(p)) * 1000))


def _assign_tier(p: float) -> str:
    for tier, lo, hi in TIER_THRESHOLDS:
        if lo <= p < hi:
            return tier
    return "Very High"


class RiskScorer:
    """
    Loads the Phase 3 best model and scores loan applications.

    Usage
    -----
    scorer = RiskScorer()
    result = scorer.score_single({"annual_income": 60000, ...})
    df_out = scorer.score_batch(applications_df)
    """

    def __init__(self, model_path: str = None):
        model_path = model_path or str(_PROJECT_ROOT / "models" / "model.pkl")
        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)
        logger.info(f"Loaded model: {type(self.model).__name__}  ({model_path})")

        self.feature_names = self._load_feature_names()
        logger.info(f"Feature set  : {len(self.feature_names)} features")

    # ── Feature name resolution ───────────────────────────────────────────────
    def _load_feature_names(self) -> list:
        train_csv = _PROJECT_ROOT / "data" / "processed" / "train.csv"
        if not train_csv.exists():
            raise FileNotFoundError(
                f"Cannot resolve feature names — {train_csv} not found.\n"
                "Run Phase 3 first."
            )
        df = pd.read_csv(train_csv, nrows=1)
        _TARGET_COLS = {
            "is_default", "default", "Default", "TARGET",
            "target", "loan_status", "bad_flag",
        }
        return [c for c in df.columns if c not in _TARGET_COLS]

    # ── Input alignment ───────────────────────────────────────────────────────
    def _align(self, df: pd.DataFrame) -> np.ndarray:
        """
        Align input columns to the exact training feature set.
        Missing columns filled with 0; extra columns ignored.
        """
        aligned = pd.DataFrame(0.0, index=df.index, columns=self.feature_names)
        for col in self.feature_names:
            if col in df.columns:
                aligned[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return aligned.values.astype(float)

    # ── Single application ────────────────────────────────────────────────────
    def score_single(self, application: dict) -> dict:
        """
        Score one application dict.
        Returns p_default, risk_score (0-1000), risk_tier, confidence.
        """
        X         = self._align(pd.DataFrame([application]))
        p_default = float(self.model.predict_proba(X)[0, 1])
        return {
            "p_default":  round(p_default, 4),
            "risk_score": _prob_to_score(p_default),
            "risk_tier":  _assign_tier(p_default),
            "confidence": round(max(p_default, 1.0 - p_default), 4),
        }

    # ── Batch scoring ─────────────────────────────────────────────────────────
    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame of applications.
        Adds columns: p_default, risk_score, risk_tier.
        """
        X      = self._align(df)
        probas = self.model.predict_proba(X)[:, 1]

        result               = df.copy()
        result["p_default"]  = probas.round(4)
        result["risk_score"] = [_prob_to_score(p) for p in probas]
        result["risk_tier"]  = [_assign_tier(p)   for p in probas]

        counts = result["risk_tier"].value_counts()
        logger.info(
            f"Scored {len(df):,} applications | "
            + " | ".join(
                f"{t}: {counts.get(t, 0)}"
                for t in ["Low", "Medium", "High", "Very High"]
            )
        )
        return result

    # ── Portfolio summary ─────────────────────────────────────────────────────
    def get_risk_distribution(self, scored_df: pd.DataFrame) -> dict:
        counts = scored_df["risk_tier"].value_counts().to_dict()
        total  = len(scored_df)
        return {
            "total":           total,
            "tier_counts":     counts,
            "tier_pct":        {k: round(v / total * 100, 2) for k, v in counts.items()},
            "mean_p_default":  round(float(scored_df["p_default"].mean()), 4),
            "mean_risk_score": round(float(scored_df["risk_score"].mean()), 1),
            "high_risk_pct":   round(
                float(scored_df["risk_tier"].isin(["High", "Very High"]).mean() * 100), 2
            ),
        }
