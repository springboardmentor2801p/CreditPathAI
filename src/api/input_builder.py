import numpy as np

def build_input(data):

    loan_amount = float(data.get("loan_amount", 0))
    income = float(data.get("income", 1))
    credit_score = float(data.get("credit_score", 650))
    ltv = float(data.get("ltv", 80))
    dtir1 = float(data.get("dtir1", 30))

    # 🔹 Derived features
    repayment_capacity = income / (loan_amount + 1)
    loan_burden_ratio = loan_amount / (income + 1)
    financial_stress_index = ltv * dtir1
    loan_amount_log = np.log1p(loan_amount)
    income_log = np.log1p(income)
    risk_combined = ltv * dtir1 / (credit_score + 1)
    financial_strength = (income * credit_score) / (loan_amount + 1)

    # 🔹 Full feature input
    return {
        "year": 2020,
        "loan_limit": "cf",
        "gender": "Male",
        "approv_in_adv": "no",
        "financial_strength": financial_strength,
        "loan_type": "type1",
        "loan_purpose": "p1",
        "credit_worthiness": "l1",
        "risk_combined": risk_combined,
        "open_credit": "nopc",
        "business_or_commercial": "no",
        "loan_amount": loan_amount,
        "term": 360,
        "neg_ammortization": "not_neg",
        "interest_only": "no",
        "lump_sum_payment": "not_lump",
        "property_value": loan_amount * 1.2,
        "construction_type": "sb",
        "occupancy_type": "pr",
        "secured_by": "home",
        "total_units": "1U",
        "income": income,
        "credit_type": "exp",
        "credit_score": credit_score,
        "co-applicant_credit_type": "exp",
        "age": 35,
        "submission_of_application": "to_inst",
        "ltv": ltv,
        "region": "North",
        "security_type": "direct",
        "dtir1": dtir1,
        "repayment_capacity": repayment_capacity,
        "loan_burden_ratio": loan_burden_ratio,
        "financial_stress_index": financial_stress_index,
        "loan_amount_log": loan_amount_log,
        "income_log": income_log
    }