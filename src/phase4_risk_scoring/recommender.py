import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Rate bands, amount caps, decisions ───────────────────────────────────────
_RATE_BANDS = {
    "Low":       {"min": 5.5,  "max": 9.0,  "rec": 6.5},
    "Medium":    {"min": 9.0,  "max": 14.0, "rec": 11.0},
    "High":      {"min": 14.0, "max": 20.0, "rec": 17.0},
    "Very High": {"min": 20.0, "max": 28.0, "rec": 24.0},
}

_AMOUNT_MULTIPLIERS = {   # × monthly income
    "Low":       10.0,
    "Medium":    6.0,
    "High":      3.0,
    "Very High": 1.5,
}

_DECISIONS = {
    "Low":       "APPROVE",
    "Medium":    "CONDITIONAL_APPROVE",
    "High":      "REFER",
    "Very High": "DECLINE",
}


@dataclass
class LoanRecommendation:
    decision:            str
    risk_tier:           str
    risk_score:          int
    p_default:           float
    interest_rate_min:   float
    interest_rate_max:   float
    interest_rate_rec:   float
    max_loan_multiplier: float
    conditions:          List[str] = field(default_factory=list)
    improvement_tips:    List[str] = field(default_factory=list)
    explanation:         str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LoanRecommender:
    """
    Converts a RiskScorer result into an actionable loan recommendation.

    Usage
    -----
    rec = LoanRecommender().recommend(scorer_result, applicant_profile)
    """

    def recommend(
        self,
        scorer_result: dict,
        applicant_profile: Optional[dict] = None,
    ) -> LoanRecommendation:
        tier  = scorer_result["risk_tier"]
        score = scorer_result["risk_score"]
        p     = scorer_result["p_default"]
        band  = _RATE_BANDS[tier]

        rec = LoanRecommendation(
            decision            = _DECISIONS[tier],
            risk_tier           = tier,
            risk_score          = score,
            p_default           = p,
            interest_rate_min   = band["min"],
            interest_rate_max   = band["max"],
            interest_rate_rec   = band["rec"],
            max_loan_multiplier = _AMOUNT_MULTIPLIERS[tier],
            conditions          = self._conditions(tier, applicant_profile),
            improvement_tips    = self._tips(tier, score, applicant_profile),
            explanation         = self._explain(tier, score, p),
        )
        logger.info(
            f"Recommendation: {rec.decision} | Tier: {tier} | "
            f"Score: {score} | Rate: {band['rec']}%"
        )
        return rec

    # ── Conditions ────────────────────────────────────────────────────────────
    def _conditions(self, tier: str, profile: Optional[dict]) -> List[str]:
        base = {
            "Low":    [],
            "Medium": [
                "Provide 3 months of recent bank statements",
                "Maintain debt-to-income ratio below 40%",
            ],
            "High": [
                "Provide 6 months of bank statements",
                "Co-signer or guarantor required",
                "Loan capped at 3x monthly income",
                "Collateral may be required for amounts above INR 2 lakh",
            ],
            "Very High": [
                "Referred for manual underwriting review",
                "Full income and asset documentation required",
                "Collateral mandatory",
                "Prior credit issues must be explained in writing",
            ],
        }.get(tier, [])

        if profile:
            cu = str(profile.get("credit_utilization_bucket", "")).lower()
            if "high" in cu or "very" in cu:
                base.append(
                    "Reduce credit card utilisation below 30% before disbursement"
                )
            if str(profile.get("home_ownership", "")).upper() == "RENT":
                base.append(
                    "Proof of stable rental history required (12+ months)"
                )
        return base

    # ── Improvement tips ─────────────────────────────────────────────────────
    def _tips(
        self, tier: str, score: int, profile: Optional[dict]
    ) -> List[str]:
        tips: List[str] = []

        if tier in ("Medium", "High", "Very High"):
            tips.append(
                f"Your risk score is {score}/1000. "
                f"Improving to 750+ unlocks lower interest rates."
            )
        if tier in ("High", "Very High"):
            tips += [
                "Pay all EMIs and credit card bills on time for 6+ months",
                "Reduce outstanding revolving credit balances",
                "Avoid applying for new credit in the next 6 months",
                "Dispute any incorrect entries on your credit report",
            ]
        if tier == "Very High":
            tips += [
                "Consider a secured loan or credit-builder product",
                "Maintain a savings buffer equal to 3 months of EMI",
            ]

        if profile:
            if "high" in str(profile.get("inquiry_intensity", "")).lower():
                tips.append(
                    "Multiple recent credit inquiries are reducing your score. "
                    "Space future applications at least 6 months apart."
                )
            ch = str(profile.get("credit_history_bucket", "")).lower()
            if "short" in ch or "new" in ch:
                tips.append(
                    "Your credit history is short. "
                    "Keeping existing accounts open will lengthen your credit age."
                )
        return tips

    # ── Plain-English explanation ─────────────────────────────────────────────
    def _explain(self, tier: str, score: int, p: float) -> str:
        pct = round(p * 100, 1)
        return {
            "Low": (
                f"Strong credit indicators — score {score}/1000, "
                f"{pct}% estimated default probability. "
                f"You qualify for our most competitive interest rates."
            ),
            "Medium": (
                f"Moderate credit risk — score {score}/1000 ({pct}% default probability). "
                f"Approval is possible with standard documentation."
            ),
            "High": (
                f"Elevated default risk — score {score}/1000 ({pct}% probability). "
                f"A co-signer or collateral can support a conditional approval."
            ),
            "Very High": (
                f"Very high credit risk — score {score}/1000 ({pct}% probability). "
                f"We recommend credit improvement before reapplying. "
                f"See tips below."
            ),
        }.get(tier, "")
