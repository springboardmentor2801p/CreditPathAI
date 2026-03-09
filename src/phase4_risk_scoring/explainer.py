# ── Warning suppression ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# SHAP is optional — falls back to feature importance if not installed
try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP available -- full explanations enabled")
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning(
        "SHAP not installed. Run:  pip install shap\n"
        "Falling back to feature-importance explanations."
    )


class RiskExplainer:
    """
    Explains individual risk predictions using SHAP (preferred)
    or feature importances (fallback).

    Usage
    -----
    explainer = RiskExplainer(model, feature_names, background_data=X_train[:200])
    result    = explainer.explain_single(X_row, output_dir, applicant_id="APP_001")
    explainer.generate_summary_plot(X_sample, output_dir)
    """

    def __init__(self, model, feature_names: list, background_data=None):
        self.model         = model
        self.feature_names = feature_names
        self._shap         = None

        if SHAP_AVAILABLE and background_data is not None:
            self._init_shap(background_data)

    def _init_shap(self, background_data):
        try:
            import xgboost  as xgb_lib
            import lightgbm as lgb_lib

            # Unwrap Ensemble to its best inner model
            inner = self.model
            if hasattr(self.model, "_base_models"):
                bm    = self.model._base_models
                inner = bm.get("LightGBM", bm.get("XGBoost", next(iter(bm.values()))))

            if isinstance(inner, (xgb_lib.XGBClassifier, lgb_lib.LGBMClassifier)):
                self._shap = shap.TreeExplainer(inner)
            else:
                bg         = shap.sample(background_data, min(100, len(background_data)))
                self._shap = shap.KernelExplainer(
                    lambda x: inner.predict_proba(x)[:, 1], bg
                )
            logger.info("SHAP explainer initialised successfully")
        except Exception as exc:
            logger.warning(f"SHAP init failed ({exc}) -- falling back to importances")
            self._shap = None

    # ── Single explanation ────────────────────────────────────────────────────
    def explain_single(
        self,
        X_row: np.ndarray,
        output_dir: str = "models/phase4_outputs/explanations",
        applicant_id: str = "applicant",
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        if SHAP_AVAILABLE and self._shap is not None:
            return self._shap_explain(X_row, output_dir, applicant_id)
        return self._importance_explain(X_row, output_dir, applicant_id)

    def _shap_explain(self, X_row, output_dir, applicant_id) -> dict:
        try:
            sv = self._shap.shap_values(X_row.reshape(1, -1))
            sv = sv[1] if isinstance(sv, list) else sv   # binary → positive class
            vals = sv[0]

            top_n  = min(10, len(vals))
            idx    = np.argsort(np.abs(vals))[::-1][:top_n]
            feats  = [self.feature_names[i] for i in idx]
            shvals = vals[idx].tolist()

            # Waterfall bar chart
            colours = ["#d32f2f" if v > 0 else "#388e3c" for v in shvals]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(top_n), shvals[::-1], color=colours[::-1])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(feats[::-1], fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value  (impact on default probability)")
            ax.set_title(
                f"Risk Explanation -- {applicant_id}\n"
                f"Red = raises default risk  |  Green = lowers default risk"
            )
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            path = os.path.join(output_dir, f"shap_{applicant_id}.png")
            plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()

            return {
                "method": "SHAP",
                "top_features": [
                    {
                        "feature":    f,
                        "shap_value": round(v, 4),
                        "direction":  "increases_risk" if v > 0 else "decreases_risk",
                    }
                    for f, v in zip(feats, shvals)
                ],
                "plot_path": path,
            }
        except Exception as exc:
            logger.warning(f"SHAP single explain failed: {exc}")
            return self._importance_explain(X_row, output_dir, applicant_id)

    def _importance_explain(self, X_row, output_dir, applicant_id) -> dict:
        imps = self._get_importances()
        if imps is None:
            return {"method": "none", "top_features": [], "plot_path": None}

        top_n = min(10, len(imps))
        idx   = np.argsort(imps)[::-1][:top_n]
        feats = [self.feature_names[i] for i in idx]
        vals  = imps[idx].tolist()
        fvals = X_row[idx].tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), vals[::-1], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats[::-1], fontsize=9)
        ax.set_xlabel("Feature Importance Score")
        ax.set_title(f"Risk Explanation -- {applicant_id}  (Feature Importance)")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        path = os.path.join(output_dir, f"importance_{applicant_id}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()

        return {
            "method": "feature_importance",
            "top_features": [
                {"feature": f, "importance": round(v, 4), "value": round(fv, 4)}
                for f, v, fv in zip(feats, vals, fvals)
            ],
            "plot_path": path,
        }

    def _get_importances(self):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "_base_models"):
            for m in self.model._base_models.values():
                if hasattr(m, "feature_importances_"):
                    return m.feature_importances_
        if hasattr(self.model, "named_steps"):
            clf = self.model.named_steps.get("clf")
            if clf is not None and hasattr(clf, "coef_"):
                return np.abs(clf.coef_[0])
        return None

    # ── SHAP summary plot (portfolio) ─────────────────────────────────────────
    def generate_summary_plot(
        self,
        X_sample: np.ndarray,
        output_dir: str = "models/phase4_outputs/explanations",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        if not (SHAP_AVAILABLE and self._shap is not None):
            logger.warning("SHAP not available -- summary plot skipped")
            return ""
        try:
            sv = self._shap.shap_values(X_sample)
            sv = sv[1] if isinstance(sv, list) else sv
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                sv, X_sample,
                feature_names=self.feature_names,
                plot_type="dot", show=False, max_display=20,
            )
            path = os.path.join(output_dir, "shap_summary_plot.png")
            plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
            logger.info(f"SHAP summary plot saved {path}")
            return path
        except Exception as exc:
            logger.warning(f"SHAP summary plot failed: {exc}")
            return ""
