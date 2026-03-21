def recommendation_engine(model, input_data):

    # Step 1: Predict default probability
    prob = model.predict_proba(input_data)[0][1]

    # Step 2: Extract loan amount
    loan_amount = float(input_data["loan_amount"].values[0])

    # Step 3: Calculate expected loss
    expected_loss = prob * loan_amount

    # Step 4: Decision logic
    if expected_loss < 50000:

        risk_level = "Low"

        decision_plan = {
            "assigned_team": "Automated System",
            "recovery_channel": "Email + SMS Reminder",
            "follow_up_frequency": "Once in 15 days",
            "legal_action": False
        }

        action = "Send automated reminder"

    elif expected_loss < 200000:

        risk_level = "Medium"

        decision_plan = {
            "assigned_team": "Call Center Agent",
            "recovery_channel": "Phone Call + EMI Restructure Offer",
            "follow_up_frequency": "Weekly",
            "legal_action": False
        }

        action = "Call borrower and discuss repayment"

    elif expected_loss < 500000:

        risk_level = "High"

        decision_plan = {
            "assigned_team": "Dedicated Recovery Officer",
            "recovery_channel": "Direct Call + Field Visit",
            "follow_up_frequency": "Every 5 days",
            "legal_action": False
        }

        action = "Escalate to recovery team"

    else:

        risk_level = "Critical"

        decision_plan = {
            "assigned_team": "Senior Recovery & Legal Team",
            "recovery_channel": "Legal Notice + Field Investigation",
            "follow_up_frequency": "Every 3 days",
            "legal_action": True
        }

        action = "Escalate to legal recovery"

    # Step 5: Return results
    return {
        "default_probability": round(float(prob), 2),
        "expected_loss": int(expected_loss),
        "risk_level": risk_level,
        "recommended_action": action,
        "decision_plan": decision_plan
    }