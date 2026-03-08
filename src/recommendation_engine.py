def generate_recommendation(probability):

    if probability > 0.8:
        return {
            "risk_level": "High Risk",
            "action": "Reject Loan or Request Collateral"
        }

    elif probability > 0.5:
        return {
            "risk_level": "Medium Risk",
            "action": "Approve with Higher Interest Rate"
        }

    else:
        return {
            "risk_level": "Low Risk",
            "action": "Approve Loan"
        }