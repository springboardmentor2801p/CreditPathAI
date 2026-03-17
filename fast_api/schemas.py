"""
================================================================================
  CreditPathAI — fast_api/schemas.py
  Purpose : Pydantic request / response models for the FastAPI recommendation server.
================================================================================
"""

from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


# ─────────────────────────── Request Schema ───────────────────────────────────

class BorrowerRequest(BaseModel):
    """
    All raw borrower features required by the recommendation engine.
    Every field maps 1-to-1 with the feature dict accepted by recommend().
    Fields that can be derived (log_*, interaction features) are computed
    internally by the engine, so the caller only needs the base features.
    """

    # Loan characteristics
    isJointApplication: int = Field(..., ge=0, le=1, description="1 if joint application, else 0")
    loanAmount: float = Field(..., gt=0, description="Loan amount in currency units")
    interestRate: float = Field(..., gt=0, description="Annual interest rate (%)")
    monthlyPayment: float = Field(..., gt=0, description="Monthly payment amount")
    term_months: int = Field(..., gt=0, description="Loan term in months")

    # Borrower profile
    yearsEmployment: float = Field(..., ge=0, description="Years of employment")
    annualIncome: float = Field(..., gt=0, description="Annual income")
    incomeVerified: int = Field(..., ge=0, le=1, description="1 if income verified, else 0")

    # Debt / credit ratios
    dtiRatio: float = Field(..., ge=0, description="Debt-to-income ratio")
    revolvingBalance: float = Field(..., ge=0, description="Total revolving credit balance")
    revolvingUtilizationRate: float = Field(..., ge=0, description="Revolving credit utilisation rate")

    # Credit history
    lengthCreditHistory: int = Field(..., ge=0, description="Length of credit history in years")
    numTotalCreditLines: int = Field(..., ge=0, description="Total number of credit lines")
    numOpenCreditLines: int = Field(..., ge=0, description="Number of currently open credit lines")
    numOpenCreditLines1Year: int = Field(..., ge=0, description="Open credit lines in the last 1 year")
    numDerogatoryRec: int = Field(..., ge=0, description="Number of derogatory records")
    numDelinquency2Years: int = Field(..., ge=0, description="Number of delinquencies in the last 2 years")
    numChargeoff1year: int = Field(..., ge=0, description="Number of charge-offs in the last year")
    numInquiries6Mon: int = Field(..., ge=0, description="Number of credit inquiries in the last 6 months")

    # Pre-engineered ratio features (as computed during training)
    grade_score: float = Field(..., description="Numeric grade score of the loan")
    loan_to_income_ratio: float = Field(..., ge=0, description="Loan amount / annual income")
    payment_to_income_ratio: float = Field(..., ge=0, description="Monthly payment / monthly income")
    repayment_velocity: float = Field(..., ge=0, description="Repayment velocity metric")
    loan_amortization_rate: float = Field(..., ge=0, description="Rate of loan amortisation")
    open_credit_ratio: float = Field(..., ge=0, description="Open credit lines / total credit lines")
    recent_credit_velocity: float = Field(..., ge=0, description="Velocity of recent credit usage")
    inquiry_intensity: float = Field(..., ge=0, description="Inquiry intensity measure")
    delinquency_density: float = Field(..., ge=0, description="Delinquency density measure")
    derogatory_density: float = Field(..., ge=0, description="Derogatory record density measure")
    estimated_credit_limit: float = Field(..., ge=0, description="Estimated total credit limit")
    credit_utilization_recomputed: float = Field(..., ge=0, description="Recomputed credit utilisation")

    # Log-transformed features
    log_loanAmount: float = Field(..., description="log1p(loanAmount)")
    log_annualIncome: float = Field(..., description="log1p(annualIncome)")
    log_revolvingBalance: float = Field(..., description="log1p(revolvingBalance)")

    # Purpose flags (one-hot)
    purpose_business: int = Field(..., ge=0, le=1)
    purpose_debtconsolidation: int = Field(..., ge=0, le=1)
    purpose_education: int = Field(..., ge=0, le=1)
    purpose_healthcare: int = Field(..., ge=0, le=1)
    purpose_homeimprovement: int = Field(..., ge=0, le=1)
    purpose_other: int = Field(..., ge=0, le=1)

    # Home ownership flags (one-hot)
    homeOwnership_own: int = Field(..., ge=0, le=1)
    homeOwnership_rent: int = Field(..., ge=0, le=1)

    # Optional scoring threshold override
    threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="P(default) cutoff for binary predicted_default flag (default: 0.50)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "isJointApplication": 0,
                "loanAmount": 350000,
                "interestRate": 19.5,
                "monthlyPayment": 9800,
                "term_months": 36,
                "yearsEmployment": 2,
                "annualIncome": 480000,
                "incomeVerified": 1,
                "dtiRatio": 0.45,
                "revolvingBalance": 120000,
                "revolvingUtilizationRate": 0.82,
                "lengthCreditHistory": 5,
                "numTotalCreditLines": 8,
                "numOpenCreditLines": 5,
                "numOpenCreditLines1Year": 2,
                "numDerogatoryRec": 1,
                "numDelinquency2Years": 3,
                "numChargeoff1year": 1,
                "numInquiries6Mon": 4,
                "grade_score": 5,
                "loan_to_income_ratio": 0.73,
                "payment_to_income_ratio": 0.245,
                "repayment_velocity": 0.028,
                "loan_amortization_rate": 0.033,
                "open_credit_ratio": 0.625,
                "recent_credit_velocity": 2,
                "inquiry_intensity": 0.67,
                "delinquency_density": 0.6,
                "derogatory_density": 0.2,
                "estimated_credit_limit": 146000,
                "credit_utilization_recomputed": 0.82,
                "log_loanAmount": 12.766,
                "log_annualIncome": 13.082,
                "log_revolvingBalance": 11.695,
                "purpose_business": 0,
                "purpose_debtconsolidation": 1,
                "purpose_education": 0,
                "purpose_healthcare": 0,
                "purpose_homeimprovement": 0,
                "purpose_other": 0,
                "homeOwnership_own": 0,
                "homeOwnership_rent": 1,
                "threshold": 0.50,
            }
        }


# ─────────────────────────── Response Schema ─────────────────────────────────

class RecommendationResponse(BaseModel):
    """Full recommendation output returned by the engine for a single borrower."""

    # ML scores
    default_probability: float = Field(..., description="Model predicted P(default) — 0 to 1")
    predicted_default: bool = Field(..., description="True if default probability >= threshold")
    risk_band: str = Field(..., description="Qualitative risk band: Very Low / Low / Medium / High / Very High")

    # Financial exposure
    loan_amount: float = Field(..., description="Loan amount (₹)")
    expected_loss: float = Field(..., description="Expected loss = P(default) × loan amount (₹)")

    # Action plan
    priority_level: str = Field(..., description="Priority tier: Low / Medium / High / Critical")
    assigned_team: str = Field(..., description="Team responsible for this account")
    recovery_channel: str = Field(..., description="Preferred recovery communication channel")
    follow_up_frequency: str = Field(..., description="How often to follow up")
    legal_action: bool = Field(..., description="Whether legal action is recommended")
    recommended_action: str = Field(..., description="Primary recommended action")
    escalation_notes: str = Field(..., description="Additional escalation guidance")

    # Qualitative flags
    risk_flags: List[str] = Field(..., description="List of detected risk red-flags")


# ─────────────────────────── Health Check Response ───────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_features: int
