from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("credit_risk_model_lgbm.pkl")


# -------------------------------
# HOME ROUTE (IMPORTANT)
# -------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to CreditPath AI API 🚀",
        "routes": {
            "Bank Risk": "/bank-risk",
            "User Risk": "/user-risk"
        }
    }


# -------------------------------
# BANK INPUT MODEL
# -------------------------------
class BankInput(BaseModel):
    loan_amount: float
    income: float
    Credit_Score: float
    LTV: float
    dtir1: float


# -------------------------------
# USER INPUT MODEL
# -------------------------------
class UserInput(BaseModel):
    loan_type: str
    income: float
    credit_score: float
    loan_amount: float
    missed_payments: int


# -------------------------------
# BANK RECOMMENDATION ENGINE
# -------------------------------
def bank_engine(data):

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]
    expected_loss = prob * data["loan_amount"]

    # Approval
    if prob < 0.3:
        approval = "Approved"
    elif prob < 0.6:
        approval = "Conditionally Approved"
    else:
        approval = "Rejected"

    # Decision
    if expected_loss < 50000:
        decision = {
            "priority": "Low",
            "recovery_channel": "Email + SMS",
            "follow_up": "15 days"
        }
    elif expected_loss < 200000:
        decision = {
            "priority": "Medium",
            "recovery_channel": "Phone Call",
            "follow_up": "Weekly"
        }
    else:
        decision = {
            "priority": "High",
            "recovery_channel": "Field Visit",
            "follow_up": "Every 5 days"
        }

    return {
        "default_probability": round(prob, 2),
        "expected_loss": round(expected_loss, 2),
        "loan_status": approval,
        "bank_decision": decision
    }


# -------------------------------
# USER RECOMMENDATION ENGINE
# -------------------------------
def user_engine(data):

    loan_type = data["loan_type"]
    credit_score = data["credit_score"]

    # Risk logic
    if credit_score > 750:
        risk = "Low"
        advice = "You are eligible. Maintain your credit score."
        tips = [
            "Maintain timely EMI payments",
            "Keep credit utilization low"
        ]

    elif credit_score > 600:
        risk = "Medium"
        advice = "Improve credit score and reduce debt before applying."
        tips = [
            "Pay EMIs on time",
            "Avoid multiple loans",
            "Reduce credit card usage"
        ]

    else:
        risk = "High"
        advice = "High risk. Avoid applying now."
        tips = [
            "Clear existing debts",
            "Avoid applying for new loans",
            "Improve repayment history"
        ]

    # Loan type suggestion
    if loan_type == "home":
        suggestion = "Long-term planning required"
    elif loan_type == "car":
        suggestion = "Short-term manageable loan"
    else:
        suggestion = "Check loan terms carefully"

    return {
        "risk_level": risk,
        "recommendation_summary": [
            advice,
            f"Loan Insight: {suggestion}",
            "Action: " + tips[0]
        ],
        "tips": tips
    }


# -------------------------------
# BANK API
# -------------------------------
import pandas as pd

@app.post("/bank-risk")
def bank_api(data: BankInput):

    try:
        input_dict = data.dict()

        # Step 1: Create empty row with 36 features
        model_features = model.feature_name_  # get all feature names

        full_data = {feature: 0 for feature in model_features}

        # Step 2: Update only known inputs
        full_data.update({
            "loan_amount": input_dict["loan_amount"],
            "income": input_dict["income"],
            "Credit_Score": input_dict["Credit_Score"],
            "LTV": input_dict["LTV"],
            "dtir1": input_dict["dtir1"]
        })

        # Step 3: Convert to DataFrame
        input_df = pd.DataFrame([full_data])

        # Step 4: Predict
        prob = model.predict_proba(input_df)[0][1]
        expected_loss = prob * input_dict["loan_amount"]

        # Step 5: Decision
        if expected_loss < 50000:
            status = "Approved"
            priority = "Low"
            channel = "Email + SMS"
            follow = "15 days"

        elif expected_loss < 200000:
            status = "Conditionally Approved"
            priority = "Medium"
            channel = "Call Center"
            follow = "Weekly"

        else:
            status = "High Risk"
            priority = "High"
            channel = "Field Visit"
            follow = "Every 5 days"

        return {
            "default_probability": round(float(prob), 2),
            "expected_loss": round(float(expected_loss), 2),
            "loan_status": status,
            "bank_decision": {
                "priority": priority,
                "recovery_channel": channel,
                "follow_up": follow
            }
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# USER API
# -------------------------------
@app.post("/user-risk")
def user_api(input: UserInput):
    return user_engine(input.dict())