def borrower_recommendation(input_data, prob):

    suggestions = []

    loan_amount = float(input_data["loan_amount"].values[0])
    income = float(input_data["income"].values[0])
    credit_score = float(input_data["Credit_Score"].values[0])
    age = float(input_data["age"].values[0])
    ltv = float(input_data["LTV"].values[0])
    dti = float(input_data["dtir1"].values[0])


    # income vs loan
    if loan_amount > income * 20:
        suggestions.append("Requested loan amount is high compared to income")

    # credit score
    if credit_score < 650:
        suggestions.append("Improve credit score above 650 for better approval")

    # high DTI
    if dti > 40:
        suggestions.append("Your EMI burden is high, reduce existing loans")

    # high LTV
    if ltv > 90:
        suggestions.append("Provide higher down payment to reduce risk")

    # age risk
    if age > 60:
        suggestions.append("Loan tenure may exceed retirement age")


    # probability based suggestions
    if prob > 0.8:
        suggestions.append(
            "High probability of default — reduce loan amount or improve income"
        )

    elif prob > 0.6:
        suggestions.append(
            "Moderate default risk — consider lowering EMI burden"
        )


    # fallback (ONLY if no suggestions triggered)
    if not suggestions:
        if prob > 0.7:
            suggestions.append(
                "High risk detected based on overall financial profile"
            )
        elif prob > 0.4:
            suggestions.append(
                "Moderate risk — improving income or reducing loan may help"
            )
        else:
            suggestions.append(
                "Profile looks good, maintain current financial discipline"
            )


    # approval label
    if prob > 0.75:
        approval = "Low"
    elif prob > 0.40:
        approval = "Medium"
    else:
        approval = "High"

    return {
        "approval_chance": approval,
        "suggestions": suggestions
    }