# ── Warning suppression ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Risk tier boundaries ───────────────────────────────────────────────────────
_TIERS = [
    ("Low",       0.00, 0.15),
    ("Medium",    0.15, 0.35),
    ("High",      0.35, 0.60),
    ("Very High", 0.60, 1.01),
]

_RATE_BANDS = {
    "Low":       {"min": 5.5,  "max": 9.0,  "rec": 6.5},
    "Medium":    {"min": 9.0,  "max": 14.0, "rec": 11.0},
    "High":      {"min": 14.0, "max": 20.0, "rec": 17.0},
    "Very High": {"min": 20.0, "max": 28.0, "rec": 24.0},
}

_DECISIONS = {
    "Low":       "APPROVE",
    "Medium":    "CONDITIONAL_APPROVE",
    "High":      "REFER",
    "Very High": "DECLINE",
}

_LOAN_MULTIPLIERS = {
    "Low": 10.0, "Medium": 6.0, "High": 3.0, "Very High": 1.5
}


def _prob_to_score(p: float) -> int:
    return int(round((1.0 - p) * 1000))


def _assign_tier(p: float) -> str:
    for tier, lo, hi in _TIERS:
        if lo <= p < hi:
            return tier
    return "Very High"


class MLInferenceEngine:
    """
    Loads the Phase 3 best model + feature names from train.csv.
    Accepts raw Borrower + Loan fields (camelCase from production schema),
    aligns them to the 108-feature training space, and returns predictions.
    """

    # Map production camelCase field names -> training snake_case column names
    # (after one-hot encoding these become prefix_value columns)
    _FIELD_MAP = {
        "residentialState":      "residential_state",
        "yearsEmployment":       "years_employment",
        "homeOwnership":         "home_ownership",
        "annualIncome":          "annual_income",
        "incomeVerified":        "income_verified",
        "dtiRatio":              "dti_ratio",
        "lengthCreditHistory":   "length_credit_history",
        "numTotalCreditLines":   "num_total_credit_lines",
        "numOpenCreditLines":    "num_open_credit_lines",
        "numOpenCreditLines1Year": "num_open_credit_lines_1year",
        "revolvingBalance":      "revolving_balance",
        "revolvingUtilizationRate": "revolving_utilization_rate",
        "numDerogatoryRec":      "num_derogatory_rec",
        "numDelinquency2Years":  "num_delinquency_2years",
        "numChargeoff1year":     "num_chargeoff_1year",
        "numInquiries6Mon":      "num_inquiries_6mon",
        # Loan fields
        "purpose":               "purpose",
        "isJointApplication":    "is_joint_application",
        "loanAmount":            "loan_amount",
        "term":                  "term",
        "interestRate":          "interest_rate",
        "monthlyPayment":        "monthly_payment",
        "grade":                 "grade",
    }

    def __init__(self):
        model_path = _PROJECT_ROOT / "models" / "model.pkl"
        train_csv  = _PROJECT_ROOT / "data" / "processed" / "train.csv"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run Phase 3 first."
            )
        if not train_csv.exists():
            raise FileNotFoundError(
                f"Feature list not found at {train_csv}. Run Phase 3 first."
            )

        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)

        _TARGETS = {"is_default", "default", "Default", "TARGET",
                    "target", "loan_status", "bad_flag"}
        df = pd.read_csv(train_csv, nrows=1)
        self.feature_names = [c for c in df.columns if c not in _TARGETS]

        logger.info(
            f"MLInferenceEngine ready — model={type(self.model).__name__}, "
            f"features={len(self.feature_names)}"
        )

    def _normalize_keys(self, raw: dict) -> dict:
        """Convert camelCase production keys to snake_case training keys."""
        normalized = {}
        for k, v in raw.items():
            normalized[self._FIELD_MAP.get(k, k)] = v
        return normalized

    def _build_feature_row(self, data: dict) -> np.ndarray:
        """
        Align a single input dict to the 108-column training feature space.
        Handles one-hot encoded columns by pattern matching.
        """
        normalized = self._normalize_keys(data)
        row = {col: 0.0 for col in self.feature_names}

        for feat_col in self.feature_names:
            # Direct numeric match
            if feat_col in normalized:
                try:
                    row[feat_col] = float(normalized[feat_col])
                except (ValueError, TypeError):
                    row[feat_col] = 0.0
                continue

            # One-hot encoded column match: e.g. "residential_state_CA"
            # Check if any normalized key is a prefix of feat_col
            for raw_key, raw_val in normalized.items():
                ohe_col = f"{raw_key}_{str(raw_val).strip().replace(' ', '_').lower()}"
                if feat_col.lower() == ohe_col.lower():
                    row[feat_col] = 1.0
                    break

        return np.array([list(row.values())], dtype=float)

    def predict(self, borrower: dict, loan: dict) -> dict:
        """
        Run inference on one application.

        Parameters
        ----------
        borrower : dict  — fields from Borrower_Prod schema (camelCase)
        loan     : dict  — fields from Loan_Prod schema (camelCase)

        Returns
        -------
        dict with p_default, risk_score, risk_tier, confidence
        """
        merged = {**borrower, **loan}
        X      = self._build_feature_row(merged)
        proba  = float(self.model.predict_proba(X)[0, 1])
        tier   = _assign_tier(proba)

        return {
            "p_default":  round(proba, 4),
            "risk_score": _prob_to_score(proba),
            "risk_tier":  tier,
            "confidence": round(max(proba, 1.0 - proba), 4),
        }

    def recommend(self, prediction: dict, borrower: dict, loan: dict) -> dict:
        """
        Build a full loan recommendation from a prediction result.
        """
        tier  = prediction["risk_tier"]
        score = prediction["risk_score"]
        p     = prediction["p_default"]
        band  = _RATE_BANDS[tier]

        # Derive max loan from annual income
        annual_income     = float(borrower.get("annualIncome", 0))
        monthly_income    = annual_income / 12.0
        max_loan_amount   = round(monthly_income * _LOAN_MULTIPLIERS[tier], 2)

        conditions = _build_conditions(tier, borrower)
        tips       = _build_tips(tier, score, borrower)

        return {
            "decision":           _DECISIONS[tier],
            "risk_tier":          tier,
            "risk_score":         score,
            "p_default":          p,
            "interest_rate_min":  band["min"],
            "interest_rate_max":  band["max"],
            "interest_rate_rec":  band["rec"],
            "max_loan_amount":    max_loan_amount,
            "conditions":         conditions,
            "improvement_tips":   tips,
            "explanation": _explain(tier, score, p),
        }


def _build_conditions(tier: str, borrower: dict) -> list:
    base = {
        "Low":    [],
        "Medium": [
            "Provide 3 months of recent bank statements",
            "Maintain DTI ratio below 40%",
        ],
        "High": [
            "Provide 6 months of bank statements",
            "Co-signer or guarantor required",
            "Collateral may be required for amounts above $20,000",
        ],
        "Very High": [
            "Referred for manual underwriting review",
            "Full income and asset documentation required",
            "Collateral mandatory",
        ],
    }.get(tier, [])

    ru = float(borrower.get("revolvingUtilizationRate", 0))
    if ru > 70:
        base.append("Reduce revolving credit utilisation below 30% before disbursement")

    ho = str(borrower.get("homeOwnership", "")).lower()
    if ho == "rent":
        base.append("Proof of stable rental history required (12+ months)")

    return base


def _build_tips(tier: str, score: int, borrower: dict) -> list:
    tips = []
    if tier in ("Medium", "High", "Very High"):
        tips.append(
            f"Your risk score is {score}/1000. Reaching 750+ unlocks lower rates."
        )
    if tier in ("High", "Very High"):
        tips += [
            "Pay all EMIs and credit cards on time for 6+ consecutive months",
            "Reduce outstanding revolving balances",
            "Avoid new credit applications for the next 6 months",
        ]
    inq = int(borrower.get("numInquiries6Mon", 0))
    if inq >= 5:
        tips.append(
            "Multiple recent credit inquiries are lowering your score. "
            "Space future applications at least 6 months apart."
        )
    derog = int(borrower.get("numDerogatoryRec", 0))
    if derog > 0:
        tips.append(
            f"You have {derog} derogatory record(s). "
            "Resolving these will significantly improve your risk profile."
        )
    return tips


def _explain(tier: str, score: int, p: float) -> str:
    pct = round(p * 100, 1)
    return {
        "Low":
            f"Strong credit profile — score {score}/1000, {pct}% default probability. "
            f"Eligible for best-rate products.",
        "Medium":
            f"Moderate risk — score {score}/1000 ({pct}% default probability). "
            f"Approval possible with standard documentation.",
        "High":
            f"Elevated risk — score {score}/1000 ({pct}% probability). "
            f"Co-signer or collateral can support conditional approval.",
        "Very High":
            f"High risk — score {score}/1000 ({pct}% probability). "
            f"Credit improvement recommended before reapplying.",
    }.get(tier, "")
