def get_recommendation(prob, data):

    loan_amount = data.get("loan_amount", 0)
    income = data.get("income", 1)
    credit_score = data.get("credit_score", 600)
    ltv = data.get("ltv", 80)
    dtir1 = data.get("dtir1", 30)

    # ================================
    # 🔹 Risk Level
    # ================================
    if prob < 0.2:
        risk = "Low"
    elif prob < 0.4:
        risk = "Medium"
    elif prob < 0.7:
        risk = "High"
    else:
        risk = "Critical"

    # ================================
    # 🔹 Behavioral Insights (with severity)
    # ================================
    insights = []

    if income < 30000:
        insights.append(("Low income reduces repayment capacity", "high"))

    if credit_score < 600:
        insights.append(("Poor credit history increases default probability", "high"))

    if ltv > 90:
        insights.append(("High loan-to-value ratio indicates low borrower equity", "medium"))

    if dtir1 > 40:
        insights.append(("High debt-to-income ratio reflects financial stress", "high"))

    if loan_amount > income * 5:
        insights.append(("Loan burden is significantly high relative to income", "high"))

    # Sort insights by severity
    insights_sorted = sorted(insights, key=lambda x: x[1], reverse=True)
    insights_text = [i[0] for i in insights_sorted]

    # ================================
    # 🔹 Intelligent Strategy Logic
    # ================================
    if risk == "Low":
        action = (
            "The borrower shows stable financial behavior. "
            "Continue automated monitoring with periodic reminders. "
            "No immediate intervention is required unless conditions change."
        )

    elif risk == "Medium":
        action = (
            "There are early signs of repayment risk. "
            "Initiate a proactive discussion with the borrower to understand potential constraints. "
            "Offer flexible repayment options such as tenure extension or temporary relief."
        )

    elif risk == "High":
        action = (
            "The borrower is likely to face repayment difficulty. "
            "Assign a recovery specialist to engage closely. "
            "Evaluate restructuring options, partial settlements, or revised repayment schedules "
            "based on borrower’s financial condition."
        )

    else:  # Critical
        action = (
            "The borrower presents a high probability of default with severe financial stress indicators. "
            "Escalate the case for immediate risk mitigation. "
            "Initiate legal review, assess collateral recovery, and implement strict collection measures."
        )

    # ================================
    # 🔹 Expected Loss
    # ================================
    expected_loss = prob * loan_amount

    # ================================
    # 🔹 Confidence Tag (NEW 🔥)
    # ================================
    if prob > 0.8:
        confidence = "Very High Risk Confidence"
    elif prob > 0.6:
        confidence = "High Risk Confidence"
    elif prob > 0.3:
        confidence = "Moderate Risk Confidence"
    else:
        confidence = "Low Risk Confidence"

    # ================================
    # 🔹 Final Response
    # ================================
    return {
        "default_probability": round(float(prob), 4),
        "risk_level": risk,
        "confidence": confidence,
        "expected_loss": round(expected_loss, 2),
        "key_risk_factors": insights_text,
        "recommended_strategy": action
    }

# TEST
if __name__ == "__main__":
    sample = {
        "loan_amount": 500000,
        "income": 20000,
        "credit_score": 500,
        "ltv": 95,
        "dtir1": 50
    }

    result = get_recommendation(0.65, sample)
    print(result)