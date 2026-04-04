import joblib
import pandas as pd

model = joblib.load("credit_risk_model.pkl")

def risk_scoring_engine(data):

    loan = data["loanAmount"]
    income = data["annualIncome"]
    dti = data["dtiRatio"]
    interest = data["interestRate"]
    existing_loans = data.get("existingLoans", 0)

    # 🔢 Risk calculation (balanced formula)
    risk_score = (
        (loan / income) * 0.4 +
        dti * 0.3 +
        (interest / 100) * 0.2 +
        existing_loans * 0.1
    )

    # 🧠 Risk level
    if risk_score < 0.3:
        risk_level = "Low"
    elif risk_score < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # 🎯 Dynamic Suggestions (same style, but changing)
    suggestions = []

    if risk_level == "Low":
        suggestions.append("Maintain stable income")
        suggestions.append("Keep debt low")
        suggestions.append("Pay EMIs on time")

    elif risk_level == "Medium":
        suggestions.append("Try to reduce your debt")
        suggestions.append("Avoid taking additional loans")
        suggestions.append("Improve your credit profile")

    elif risk_level == "High":
        suggestions.append("Avoid taking large loans")
        suggestions.append("Focus on clearing existing debts")
        suggestions.append("Improve income stability")

    # 🔍 Extra intelligent conditions
    if loan > income * 0.5:
        suggestions.append("Loan amount is high compared to your income")

    if dti > 0.4:
        suggestions.append("Your DTI is high, reduce liabilities")

    if interest > 12:
        suggestions.append("Interest rate is high, try negotiating")

    if existing_loans > 0:
        suggestions.append("Clear existing loans before applying")

    # 👤 USER VIEW
    user_view = {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "suggestions": suggestions
    }
    # 🏦 BANK VIEW LOGIC
    if existing_loans > 1:
        decision = "Reject Loan - Too many existing loans"
    elif risk_score < 0.4:
        decision = "Approve Loan"
    elif risk_score < 0.7:
        decision = "Review Required"
    else:
        decision = "Reject Loan - High Risk"

    bank_view = {
        "decision": decision,
        "risk_score": round(risk_score, 2),
        "expected_loss": int(loan * risk_score)
    }

    return {
        "user": user_view,
        "bank": bank_view
    }