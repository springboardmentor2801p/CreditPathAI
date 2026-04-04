import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CreditPathAI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ──────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "data/models/best_model.pkl")
FEATURES_PATH = "data/reports/best_model_info.json"

model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = joblib.load("data/models/feature_columns.pkl")

with open(FEATURES_PATH) as f:
    model_info = json.load(f)

explainer = shap.TreeExplainer(model)


# ── Input Schema ─────────────────────────────────────────────
class LoanInput(BaseModel):
    Credit_Score: float
    loan_amount: float
    income: float
    LTV: float
    dtir1: float
    property_value: float = None   # ✅ None so fallback triggers in engineer_features
    loan_type: int = 1
    loan_purpose: int = 1
    credit_worthiness: int = 1
    business_or_commercial: int = 0
    neg_ammortization: int = 0
    lump_sum_payment: int = 0
    occupancy_type: int = 1
    total_units: int = 1
    loan_limit: int = 0
    gender: int = 1
    credit_type: int = 1
    age: int = 35
    region: int = 1
    security_type: int = 1
    approv_in_adv: int = 0
    open_credit: int = 1
    term: int = 360
    interest_only: int = 0
    construction_type: int = 1
    secured_by: int = 1


# ── Feature Engineering ──────────────────────────────────────
def engineer_features(data: dict) -> pd.DataFrame:
    loan_amount  = data["loan_amount"]
    income       = data["income"]
    credit_score = data["Credit_Score"]
    ltv          = data["LTV"]
    dtir1        = data["dtir1"]

    # ✅ None fallback works correctly now
    property_value = data.get("property_value") or loan_amount * 1.2

    ltv_band            = int(min(ltv // 20, 4))
    income_loan_ratio   = income / loan_amount if loan_amount > 0 else 0
    property_loan_gap   = property_value - loan_amount
    loan_to_value_ratio = loan_amount / property_value if property_value > 0 else 0

    credit_risk_score = 0
    if credit_score < 600:   credit_risk_score += 3
    elif credit_score < 680: credit_risk_score += 2
    elif credit_score < 740: credit_risk_score += 1
    if ltv > 90:   credit_risk_score += 3
    elif ltv > 80: credit_risk_score += 2
    elif ltv > 70: credit_risk_score += 1
    if dtir1 > 50:   credit_risk_score += 2
    elif dtir1 > 43: credit_risk_score += 1
    high_risk_flag = int(credit_risk_score >= 5)

    if dtir1 < 28:   debt_to_income_band = 0
    elif dtir1 < 36: debt_to_income_band = 1
    elif dtir1 < 43: debt_to_income_band = 2
    else:            debt_to_income_band = 3

    loan_amount_log         = np.log1p(loan_amount)
    income_log              = np.log1p(income)
    property_value_log      = np.log1p(property_value)
    loan_income_interaction = loan_amount_log * income_log
    credit_ltv_interaction  = credit_score * ltv

    age = data.get("age", 35)
    if age < 25:   age_band = 0
    elif age < 35: age_band = 1
    elif age < 50: age_band = 2
    elif age < 65: age_band = 3
    else:          age_band = 4

    loan_purpose_risk = {1: 1, 2: 2, 3: 3, 4: 2}.get(data.get("loan_purpose", 1), 1)
    credit_type_risk  = {1: 1, 2: 2, 3: 3}.get(data.get("credit_type", 1), 1)

    # ✅ Only the 33 features the model was actually trained on
    row = {
        "loan_limit":                data.get("loan_limit", 0),
        "gender":                    data.get("gender", 1),
        "approv_in_adv":             data.get("approv_in_adv", 0),
        "loan_type":                 data.get("loan_type", 1),
        "loan_purpose":              data.get("loan_purpose", 1),
        "credit_worthiness":         data.get("credit_worthiness", 1),
        "open_credit":               data.get("open_credit", 1),
        "business_or_commercial":    data.get("business_or_commercial", 0),
        "loan_amount":               loan_amount,
        "term":                      data.get("term", 360),
        "neg_ammortization":         data.get("neg_ammortization", 0),
        "interest_only":             data.get("interest_only", 0),
        "lump_sum_payment":          data.get("lump_sum_payment", 0),
        "property_value":            property_value,
        "construction_type":         data.get("construction_type", 1),
        "occupancy_type":            data.get("occupancy_type", 1),
        "secured_by":                data.get("secured_by", 1),
        "total_units":               data.get("total_units", 1),
        "income":                    income,
        "credit_type":               data.get("credit_type", 1),
        "credit_score":              credit_score,
        "co-applicant_credit_type":  data.get("credit_type", 1),
        "age":                       age,
        "submission_of_application": 1,
        "ltv":                       ltv,
        "region":                    data.get("region", 1),
        "security_type":             data.get("security_type", 1),
        "dtir1":                     dtir1,
        "ltv_band":                  ltv_band,
        "income_loan_ratio":         round(income_loan_ratio, 4),
        "credit_risk_score":         credit_risk_score,
        "property_loan_gap":         round(property_loan_gap, 2),
        "high_risk_flag":            high_risk_flag,
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ── Helpers ──────────────────────────────────────────────────
def get_risk_level(prob: float, credit_score: float) -> str:
    # Spec §3.3.3 — based on default probability
    if credit_score < 500:
        return "CRITICAL"
    if prob < 0.15:   return "LOW"
    elif prob < 0.40: return "MEDIUM"
    elif prob < 0.60: return "HIGH"
    else:             return "CRITICAL"


def get_eligibility_status(default_prob: float) -> str:
    # Spec §3.3.4 — based on default probability
    if default_prob < 0.15: return "APPROVED"
    if default_prob < 0.40: return "CONDITIONAL"
    return "NOT_APPROVED"


# ── Rule-based override (handles model mispredictions) ───────
def apply_rule_override(approval_prob: float, cs: float, ltv: float, dti: float) -> float:
    # ── REJECTED: clearly bad profile ────────────────────────
    hard_fail = (cs < 550 and ltv > 85) or (dti > 50 and cs < 600) or cs < 450
    if hard_fail and approval_prob > 0.5:
        return 0.05

    # ── CONDITIONAL: borderline profile ──────────────────────
    # 2+ risk flags → cap approval at 0.54 so default_prob = 0.46 → CONDITIONAL band
    risk_flags = 0
    if cs <= 680:  risk_flags += 1   # <= catches cs=680 exactly
    if ltv > 80:   risk_flags += 1
    if dti > 36:   risk_flags += 1

    if risk_flags >= 2 and approval_prob > 0.78:
        return 0.78  # default_prob = 0.22 → CONDITIONAL band (0.15–0.40 per spec)

    return approval_prob


# ── Basic Endpoints ──────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "CreditPathAI API v2", "model": model_info["model_name"], "roc_auc": model_info["roc_auc"]}

@app.get("/health")
def health():
    return {"status": "ok", "model": model_info["model_name"]}


# ── BANK ENDPOINT (/bank-recommendation) ────────────────────
@app.post("/bank-recommendation")
def bank_recommendation(input: LoanInput):
    try:
        data = input.model_dump()
        df   = engineer_features(data)

        # ✅ class 1 = Approved, so default_prob = 1 - predict_proba[0][1]
        raw_prob      = float(model.predict_proba(df)[0][1])
        approval_prob = round(raw_prob, 4)

        cs  = data["Credit_Score"]
        ltv = data["LTV"]
        dti = data["dtir1"]

        # ✅ Apply same rule override as applicant endpoint for consistency
        approval_prob = apply_rule_override(approval_prob, cs, ltv, dti)
        default_prob  = round(1 - approval_prob, 4)

        risk_level    = get_risk_level(default_prob, cs)
        expected_loss = default_prob * data["loan_amount"]

        risk_map = {
            # Spec §3.3.3: risk_level → (team, channel, follow_up, legal, rate_adj)
            "LOW":      ("Standard Team",    "Monitoring Only",        "Quarterly", False,  0.00),
            "MEDIUM":   ("Risk Team",        "Early Intervention",     "Monthly",   False,  1.50),
            "HIGH":     ("Senior Risk Team", "Collection Agency",      "Bi-Weekly", False,  3.00),
            "CRITICAL": ("Legal & Recovery", "Legal Action + Court",   "Weekly",    True,   3.50),
        }
        team, channel, follow_up, legal, rate_adj = risk_map[risk_level]

        # Spec §3.3.3 approval status
        if risk_level == "LOW":      approval_status = "AUTO_APPROVE"
        elif risk_level == "MEDIUM": approval_status = "APPROVE WITH CONDITIONS"
        else:                        approval_status = "DECLINE"

        insights = []
        if cs < 600:
            insights.append(f"Credit score {cs} is critically low — high default correlation.")
        elif cs < 680:
            insights.append(f"Credit score {cs} is below preferred threshold of 680.")
        else:
            insights.append(f"Credit score {cs} is within acceptable range.")

        if ltv > 90:
            insights.append(f"LTV of {ltv}% is very high — minimal equity cushion.")
        elif ltv > 80:
            insights.append(f"LTV of {ltv}% exceeds the 80% safe threshold.")
        else:
            insights.append(f"LTV of {ltv}% is within safe limits.")

        if dti > 50:
            insights.append(f"DTI of {dti}% is dangerously high — severe repayment risk.")
        elif dti > 36:
            insights.append(f"DTI of {dti}% exceeds the recommended 36% limit.")
        else:
            insights.append(f"DTI of {dti}% is within healthy range.")

        return {
            "recommendation": {
                "default_probability":      default_prob,
                "risk_level":               risk_level,
                "expected_loss":            round(expected_loss, 2),
                "assigned_team":            team,
                "recovery_channel":         channel,
                "follow_up_frequency":      follow_up,
                "legal_action_required":    legal,
                "insights":                 insights,
                "approval_status":          approval_status,
                "interest_rate_adjustment": rate_adj,
            }
        }

    except Exception as e:
        logger.error(f"Bank prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── APPLICANT ENDPOINT (/applicant-recommendation) ───────────
@app.post("/applicant-recommendation")
def applicant_recommendation(input: LoanInput):
    try:
        data = input.model_dump()
        df   = engineer_features(data)

        # ✅ class 1 = Approved
        raw_prob      = float(model.predict_proba(df)[0][1])
        approval_prob = round(raw_prob, 4)           # e.g. 0.957
        default_prob  = round(1 - raw_prob, 4)       # e.g. 0.043

        # ✅ Rule-based safety override for obvious bad borrowers
        cs   = data["Credit_Score"]
        ltv  = data["LTV"]
        dti  = data["dtir1"]
        loan = data["loan_amount"]
        inc  = data["income"]

        approval_prob = apply_rule_override(approval_prob, cs, ltv, dti)
        default_prob  = round(1 - approval_prob, 4)

        # ✅ Eligibility based on default_prob
        eligibility_status = get_eligibility_status(default_prob)

        # Headline
        if eligibility_status == "APPROVED":
            headline = "🎉 Great news — you're likely eligible!"
        elif eligibility_status == "CONDITIONAL":
            headline = "⚠️ You may qualify with some improvements"
        else:
            headline = "❌ Not eligible right now — but here's your path forward"

        # Summary
        summary = (
            f"Based on your credit score of {cs}, LTV of {ltv}%, and DTI of {dti}%, "
            f"our model estimates a {round(approval_prob * 100, 1)}% approval probability. "
        )
        if eligibility_status == "APPROVED":
            summary += "Your financial profile meets standard lending criteria."
        elif eligibility_status == "CONDITIONAL":
            summary += "Some factors need improvement before a confident approval."
        else:
            summary += "Several key metrics are outside acceptable thresholds."

        # Timeline — spec §3.3.4
        if eligibility_status == "APPROVED":
            timeline     = "N/A"
            timeline_msg = "You can apply now. Processing typically takes 7–14 business days."
        elif eligibility_status == "CONDITIONAL":
            timeline     = "3 months"
            timeline_msg = "Meet the conditions below and reapply within 3 months."
        else:
            timeline     = "6 months"
            timeline_msg = "Work on the improvements below for at least 6 months before reapplying."

        # Improvement opportunities
        improvements = []

        if cs < 700:
            gap = 700 - cs
            improvements.append({
                "area":     "Credit Score",
                "priority": "CRITICAL" if cs < 580 else "HIGH" if cs < 640 else "MEDIUM",
                "current":  str(cs),
                "target":   "700+",
                "gap":      f"+{gap} points needed",
                "timeline": "6–12 months",
                "actions":  [
                    "Pay all bills on time — payment history is 35% of your score.",
                    "Keep credit card utilisation below 30%.",
                    "Avoid opening new credit accounts in the next 6 months.",
                    "Dispute any errors on your credit report.",
                ]
            })

        if ltv > 80:
            improvements.append({
                "area":     "Loan-to-Value Ratio",
                "priority": "HIGH" if ltv > 90 else "MEDIUM",
                "current":  f"{ltv}%",
                "target":   "≤ 80%",
                "gap":      f"Need {round(ltv - 80, 1)}% more equity",
                "timeline": "3–6 months",
                "actions":  [
                    "Increase your down payment to lower the LTV.",
                    "Consider a less expensive property.",
                    "Wait for property value appreciation.",
                ]
            })

        if dti > 36:
            improvements.append({
                "area":     "Debt-to-Income Ratio",
                "priority": "CRITICAL" if dti > 50 else "HIGH" if dti > 43 else "MEDIUM",
                "current":  f"{dti}%",
                "target":   "≤ 36%",
                "gap":      f"Reduce debt by ~{round((dti - 36) / 100 * inc):,}",
                "timeline": "6–12 months",
                "actions":  [
                    "Pay off high-interest debts first (avalanche method).",
                    "Consolidate multiple debts into a single lower-rate loan.",
                    "Avoid taking on new debt obligations.",
                    "Increase income through a side income or raise.",
                ]
            })

        # ✅ More realistic income threshold: monthly EMI affordability rule
        monthly_emi            = loan * 0.007
        required_annual_income = (monthly_emi / 0.43) * 12
        if inc < required_annual_income:
            improvements.append({
                "area":     "Income vs Loan Amount",
                "priority": "MEDIUM",
                "current":  f"₹{inc:,.0f} / yr",
                "target":   f"₹{required_annual_income:,.0f} / yr",
                "gap":      f"₹{max(0, required_annual_income - inc):,.0f} income gap",
                "timeline": "6–12 months",
                "actions":  [
                    "Consider a smaller loan amount.",
                    "Add a co-applicant with additional income.",
                    "Document all income sources (freelance, rental, etc.).",
                ]
            })

        # Next steps — spec §3.3.4
        if eligibility_status == "APPROVED":
            next_steps = [
                "Gather documents: ID proof, income statements, bank statements (6 months).",
                "Get a formal valuation of the property.",
                "Submit your loan application online or at a branch.",
                "Track application status and respond to any queries promptly.",
            ]
        elif eligibility_status == "CONDITIONAL":
            next_steps = [
                "Address the improvement areas listed below.",
                "Gather supporting documents for your income and assets.",
                "Speak to a loan officer about conditional approval requirements.",
                "Reapply in 3 months once conditions are met.",
            ]
        else:
            next_steps = [
                "Download your credit report and check for errors.",
                "Set up automatic payments for all existing debts.",
                "Open a savings account dedicated to your down payment.",
                "Set a calendar reminder to reapply in 6 months.",
            ]

        # ✅ FIXED: actually return the response (was missing return keyword before)
        return {
            "recommendation": {
                "eligibility_status":        eligibility_status,
                "approval_probability":      approval_prob,    # ✅ correct
                "default_probability":       default_prob,     # ✅ correct (1 - approval)
                "headline":                  headline,
                "summary":                   summary,
                "improvement_opportunities": improvements,
                "reapplication_timeline":    timeline,
                "timeline_message":          timeline_msg,
                "next_steps":               next_steps,
            }
        }

    except Exception as e:
        logger.error(f"Applicant prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Legacy endpoints ─────────────────────────────────────────
@app.post("/predict/applicant")
def predict_applicant_legacy(input: LoanInput):
    return applicant_recommendation(input)

@app.post("/predict/bank")
def predict_bank_legacy(input: LoanInput):       # ✅ fixed syntax error
    return bank_recommendation(input)