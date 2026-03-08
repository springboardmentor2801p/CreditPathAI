from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --------------------------------------------------
# Create FastAPI App
# --------------------------------------------------
app = FastAPI(title="Credit Risk Scoring API")

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
model = joblib.load("final_model.pkl")

# --------------------------------------------------
# Input Schema (must match training features)
# --------------------------------------------------
class Borrower(BaseModel):
    monthly_income: float
    monthly_expense: float
    age: int
    has_smartphone: int
    has_wallet: int
    avg_wallet_balance: float
    on_time_payment_ratio: float
    num_loans_taken: int
    loan_to_income_ratio: float
    expense_ratio: float

# --------------------------------------------------
# Home Route
# --------------------------------------------------
@app.get("/")
def home():
    return {"message": "Credit Risk API is running successfully"}

# --------------------------------------------------
# Prediction Route
# --------------------------------------------------
@app.post("/predict")
def predict(data: Borrower):

    input_data = np.array([[
        data.monthly_income,
        data.monthly_expense,
        data.age,
        data.has_smartphone,
        data.has_wallet,
        data.avg_wallet_balance,
        data.on_time_payment_ratio,
        data.num_loans_taken,
        data.loan_to_income_ratio,
        data.expense_ratio
    ]])

    probability = model.predict_proba(input_data)[0][1]

    # Simple recommendation engine
    if probability > 0.7:
        action = "High Risk - Offer small loan with monitoring"
        risk_level = "High"
    elif probability > 0.4:
        action = "Medium Risk - Provide limited credit"
        risk_level = "Medium"
    else:
        action = "Low Risk - Eligible for normal loan"
        risk_level = "Low"

    return {
        "default_probability": float(probability),
        "risk_level": risk_level,
        "recommended_action": action
    }