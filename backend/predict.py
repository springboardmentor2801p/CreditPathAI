
# ===============================
# PREDICTION & RECOMMENDATION
# CreditPath AI
# ===============================

import pandas as pd
import numpy as np
import joblib

def load_model(model_path="creditpath_model.pkl", scaler_path="creditpath_scaler.pkl"):
    """Load saved model and scaler"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully!")
    return model, scaler

def prepare_input(credit_score, loan_amount, income, ltv, dtir1):
    """Prepare input features - exactly 23 features matching the model"""
    input_df = pd.DataFrame([{
        "Credit_Score": credit_score,
        "loan_amount": loan_amount,
        "income": income,
        "LTV": ltv,
        "dtir1": dtir1,
        "loan_to_income_ratio": loan_amount / (income + 1),
        "payment_to_income_ratio": loan_amount / (income + 1),
        "high_dti_flag": 1 if dtir1 > 40 else 0,
        "low_credit_flag": 1 if credit_score < 650 else 0,
        "property_value": loan_amount * 1.2,
        "term": 360,
        "age": 35,
        "Neg_ammortization": 0,
        "occupancy_type": 1,
        "Secured_by": 1,
        "total_units": 1,
        "submission_of_application": 1,
        "Region": 1,
        "Security_Type": 1,
        "business_or_commercial": 0,
        "open_credit": 0,
        "construction_type": 1,
        "lump_sum_payment": 0,
    }])
    return input_df

def predict_default(model, scaler, input_df):
    """Predict default probability"""
    input_scaled = scaler.transform(input_df)
    prob = model.predict_proba(input_scaled)[0][1]
    return round(float(prob), 4)

def classify_risk(prob):
    """Classify borrower risk level"""
    if prob < 0.20:
        return "Very Low Risk"
    elif prob < 0.40:
        return "Low Risk"
    elif prob < 0.60:
        return "Moderate Risk"
    elif prob < 0.80:
        return "High Risk"
    else:
        return "Critical Risk"

def expected_loss_engine(prob, loan_amount):
    """Calculate expected loss and recovery decision"""
    expected_loss = prob * loan_amount

    if expected_loss < 50000:
        decision = {
            "priority": "Low",
            "assigned_team": "Automated System",
            "recovery_channel": "Email + SMS Reminder",
            "follow_up_frequency": "Once in 15 days",
            "legal_action": False
        }
    elif expected_loss < 200000:
        decision = {
            "priority": "Medium",
            "assigned_team": "Call Center Agent",
            "recovery_channel": "Phone Call + EMI Restructure",
            "follow_up_frequency": "Weekly",
            "legal_action": False
        }
    elif expected_loss < 500000:
        decision = {
            "priority": "High",
            "assigned_team": "Dedicated Recovery Officer",
            "recovery_channel": "Legal Notice + Field Visit",
            "follow_up_frequency": "Daily",
            "legal_action": True
        }
    else:
        decision = {
            "priority": "Critical",
            "assigned_team": "Senior Recovery Team",
            "recovery_channel": "Legal Action + Court Proceedings",
            "follow_up_frequency": "Immediate",
            "legal_action": True
        }

    decision["expected_loss"] = round(float(expected_loss), 2)
    return decision

def generate_recommendation(prob, loan_amount):
    """Generate full recommendation for a borrower"""
    risk_level = classify_risk(prob)
    decision = expected_loss_engine(prob, loan_amount)

    recommendation = {
        "default_probability": prob,
        "risk_level": risk_level,
        "expected_loss": decision["expected_loss"],
        "priority": decision["priority"],
        "assigned_team": decision["assigned_team"],
        "recovery_channel": decision["recovery_channel"],
        "follow_up_frequency": decision["follow_up_frequency"],
        "legal_action": decision["legal_action"]
    }

    return recommendation

if __name__ == "__main__":
    model, scaler = load_model()

    input_df = prepare_input(
        credit_score=670,
        loan_amount=200000,
        income=60000,
        ltv=80,
        dtir1=35
    )

    prob = predict_default(model, scaler, input_df)
    result = generate_recommendation(prob, 200000)

    print("\nPrediction Result:")
    for key, val in result.items():
        print(f"  {key}: {val}")
