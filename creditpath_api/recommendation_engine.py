def risk_scoring_engine(data):

    income = data["income"]
    credit_score = data["credit_score"]
    debt_ratio = data["debt_ratio"]
    loan_amount = data["loan_amount"]
    missed_payments = data["missed_payments"]

    # simple risk calculation
    risk_score = 0

    if credit_score < 600:
        risk_score += 0.3

    if debt_ratio > 0.4:
        risk_score += 0.3

    if missed_payments > 1:
        risk_score += 0.2

    if loan_amount > income * 5:
        risk_score += 0.2

    default_probability = round(risk_score, 2)
    expected_loss = round(default_probability * loan_amount, 2)

    # recommendation logic
    if default_probability > 0.7:
        priority = "High"
        action = "Assign recovery officer"
    elif default_probability > 0.4:
        priority = "Medium"
        action = "Manual review required"
    else:
        priority = "Low"
        action = "Approve loan"

    return {
        "default_probability": default_probability,
        "expected_loss": expected_loss,
        "priority": priority,
        "recommended_action": action
    }