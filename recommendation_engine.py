# expected_loss_engine.py
import joblib
import pandas as pd

def expected_loss_engine(input_data, model):
    """
    Calculate expected loss for a loan and determine recovery strategy.

    Parameters
    ----------
    input_data : pandas.DataFrame or dict-like
        Input data containing borrower features including 'loan_amount'.
    model : trained ML model
        Model that supports predict_proba().

    Returns
    -------
    dict
        Dictionary containing default probability, expected loss,
        and recommended recovery decision plan.
    """

    # Step 1: Predict probability of default
    prob = model.predict_proba(input_data)[0][1]

    # Step 2: Get loan exposure
    loan_amount = input_data["loanAmount"].iloc[0]

    # Step 3: Calculate expected loss
    expected_loss = loan_amount - (prob * loan_amount)

    # Step 4: Decide recovery strategy based on expected loss
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
            "recovery_channel": "Phone Call + EMI Restructure Offer",
            "follow_up_frequency": "Weekly",
            "legal_action": False
        }

    elif expected_loss < 500000:
        decision = {
            "priority": "High",
            "assigned_team": "Dedicated Recovery Officer",
            "recovery_channel": "Direct Call + Field Visit",
            "follow_up_frequency": "Every 5 days",
            "legal_action": False
        }

    else:
        decision = {
            "priority": "Critical",
            "assigned_team": "Senior Recovery & Legal Team",
            "recovery_channel": "Legal Notice + Field Investigation",
            "follow_up_frequency": "Every 3 days",
            "legal_action": True
        }

    return {
        "current_probability": float(prob),
        "expected_loss": float(expected_loss),
        "decision_plan": decision
    }

def prepare_input(input_dict):
    df = pd.DataFrame([input_dict])
    return df

input_schema = ['purpose', 'isJointApplication', 'loanAmount', 'term', 'interestRate',
       'monthlyPayment', 'grade', 'loanStatus', 'residentialState',
       'yearsEmployment', 'homeOwnership', 'annualIncome', 'incomeVerified',
       'dtiRatio', 'lengthCreditHistory', 'numTotalCreditLines',
       'numOpenCreditLines', 'numOpenCreditLines1Year', 'revolvingBalance',
       'revolvingUtilizationRate', 'numDerogatoryRec', 'numDelinquency2Years',
       'numChargeoff1year', 'numInquiries6Mon']

# input_data = {}
# for col in input_schema:
#     input_data[col] = input(f"Enter {col}: ")
input_data = {
    "purpose": "debtconsolidation",
    "isJointApplication": 0,
    "loanAmount": 400000,
    "term": "36 months",
    "interestRate": 15,
    "monthlyPayment": 13500,
    "grade": "E3",
    "residentialState": "CA",
    "yearsEmployment": "10+ years",
    "homeOwnership": "mortgage",
    "annualIncome": 200000,
    "incomeVerified": 0,
    "dtiRatio": 18.4,
    "lengthCreditHistory": 12,
    "numTotalCreditLines": 22,
    "numOpenCreditLines": 8,
    "numOpenCreditLines1Year": 3,
    "revolvingBalance": 75000,
    "revolvingUtilizationRate": 42.5,
    "numDerogatoryRec": 0,
    "numDelinquency2Years": 1,
    "numChargeoff1year": 0,
    "numInquiries6Mon": 2
}

model = joblib.load(r"D:\jai\Python-Workspace\Credit-Path-AI\models\model_pipeline.pkl")
input_df = prepare_input(input_data)
result = expected_loss_engine(input_df, model)
print(result)