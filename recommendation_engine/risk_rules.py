"""
================================================================================
  CreditPathAI — recommendation_engine/risk_rules.py
  Purpose : All risk-level logic and resolution rules.
            No model code here — just policy decisions.

  Contains:
    - LOSS_BANDS         : Expected-loss → priority tier mapping
    - ACTION_PLAYBOOKS   : Priority tier → full action plan
    - PROB_RISK_BANDS    : Default probability → risk band label
    - Risk flag thresholds
    - resolve_priority() : Determine priority from expected loss
    - get_action_plan()  : Return action plan dict for a priority
    - prob_to_risk_band(): Probability → qualitative band
    - detect_risk_flags(): Rule-based red flags for a borrower
================================================================================
"""

# ─────────────────────────── Expected-Loss Priority Bands ────────────────────
# (upper_limit_exclusive, priority_label)
# Borrowers are assigned the first tier whose upper limit exceeds their EL.

LOSS_BANDS = [
    (50_000,       "Low"),
    (200_000,      "Medium"),
    (500_000,      "High"),
    (float("inf"), "Critical"),
]


# ─────────────────────────── Action Playbooks ────────────────────────────────
# Full resolution plan per priority tier.

ACTION_PLAYBOOKS: dict[str, dict] = {
    "Low": {
        "priority":            "Low",
        "assigned_team":       "Automated System",
        "recovery_channel":    "Email + SMS Reminder",
        "follow_up_frequency": "Once in 15 days",
        "legal_action":        False,
        "recommended_action":  "Send automated reminder",
        "escalation_notes":    "Monitor for 30 days; escalate if no response.",
    },
    "Medium": {
        "priority":            "Medium",
        "assigned_team":       "Call Centre Agent",
        "recovery_channel":    "Phone Call + EMI Restructure Offer",
        "follow_up_frequency": "Weekly",
        "legal_action":        False,
        "recommended_action":  "Call borrower and discuss repayment plan",
        "escalation_notes":    "Offer restructuring within 14 days; escalate if rejected.",
    },
    "High": {
        "priority":            "High",
        "assigned_team":       "Dedicated Recovery Officer",
        "recovery_channel":    "Direct Call + Field Visit",
        "follow_up_frequency": "Every 5 days",
        "legal_action":        False,
        "recommended_action":  "Assign dedicated recovery agent",
        "escalation_notes":    "Field visit within 7 days; initiate legal prep if unresponsive.",
    },
    "Critical": {
        "priority":            "Critical",
        "assigned_team":       "Senior Recovery & Legal Team",
        "recovery_channel":    "Legal Notice + Field Investigation",
        "follow_up_frequency": "Every 3 days",
        "legal_action":        True,
        "recommended_action":  "Escalate to senior recovery and legal team",
        "escalation_notes":    "Issue legal notice immediately; freeze assets if warranted.",
    },
}


# ─────────────────────────── Probability → Risk Band ─────────────────────────
# (upper_limit_exclusive, band_label)

PROB_RISK_BANDS = [
    (0.20, "Very Low"),
    (0.40, "Low"),
    (0.60, "Medium"),
    (0.80, "High"),
    (1.01, "Very High"),
]


# ─────────────────────────── Risk-Flag Thresholds ────────────────────────────

HIGH_DTI_THRESHOLD              = 0.40   # dtiRatio
HIGH_INTEREST_THRESHOLD         = 18.0   # interestRate (%)
HIGH_DELINQUENCY_THRESHOLD      = 2      # numDelinquency2Years
HIGH_INQUIRY_THRESHOLD          = 3      # numInquiries6Mon
HIGH_PAYMENT_PRESSURE_THRESHOLD = 0.35   # monthlyPayment / (annualIncome / 12)
HIGH_REVOLVING_UTIL_THRESHOLD   = 0.75   # revolvingUtilizationRate


# ─────────────────────────── Functions ───────────────────────────────────────

def resolve_priority(expected_loss: float) -> str:
    """Return the priority tier label for the given expected loss (₹)."""
    for upper, label in LOSS_BANDS:
        if expected_loss < upper:
            return label
    return "Critical"


def get_action_plan(priority: str) -> dict:
    """Return a copy of the full action plan for the given priority tier."""
    return dict(ACTION_PLAYBOOKS[priority])


def prob_to_risk_band(prob: float) -> str:
    """Map a default probability (0–1) to a qualitative risk band label."""
    for upper, label in PROB_RISK_BANDS:
        if prob < upper:
            return label
    return "Very High"


def detect_risk_flags(borrower: dict) -> list[str]:
    """
    Return a list of human-readable risk flags for a single borrower.
    All applicable flags are returned (additive).

    Parameters
    ----------
    borrower : dict with raw borrower features
    """
    flags = []

    dti = borrower.get("dtiRatio", 0)
    if dti >= HIGH_DTI_THRESHOLD:
        flags.append(f"High debt-to-income ratio ({dti:.2f} ≥ {HIGH_DTI_THRESHOLD})")

    rate = borrower.get("interestRate", 0)
    if rate >= HIGH_INTEREST_THRESHOLD:
        flags.append(f"High interest rate ({rate:.1f}% ≥ {HIGH_INTEREST_THRESHOLD}%)")

    delinq = borrower.get("numDelinquency2Years", 0)
    if delinq >= HIGH_DELINQUENCY_THRESHOLD:
        flags.append(f"Recent delinquencies ({int(delinq)} in last 2 years)")

    inq = borrower.get("numInquiries6Mon", 0)
    if inq >= HIGH_INQUIRY_THRESHOLD:
        flags.append(f"Multiple recent credit inquiries ({int(inq)} in 6 months)")

    # Compute payment pressure if not already in the dict
    monthly_income = (borrower.get("annualIncome", 1) / 12) + 1e-6
    payment_pressure = borrower.get("payment_pressure",
                                    borrower.get("monthlyPayment", 0) / monthly_income)
    if payment_pressure >= HIGH_PAYMENT_PRESSURE_THRESHOLD:
        flags.append(
            f"High payment pressure "
            f"(monthly payment is {payment_pressure * 100:.1f}% of monthly income)"
        )

    util = borrower.get("revolvingUtilizationRate", 0)
    if util >= HIGH_REVOLVING_UTIL_THRESHOLD:
        flags.append(f"High revolving utilisation ({util * 100:.1f}%)")

    derog = borrower.get("numDerogatoryRec", 0)
    if derog > 0:
        flags.append(f"Has derogatory records ({int(derog)})")

    chargeoff = borrower.get("numChargeoff1year", 0)
    if chargeoff > 0:
        flags.append(f"Charge-offs in past year ({int(chargeoff)})")

    return flags
