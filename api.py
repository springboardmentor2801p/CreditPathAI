from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
app = FastAPI(title="CreditPathAI Risk Scoring API")
model = joblib.load("lightgbm_model.pkl")
model_columns = joblib.load("model_columns.pkl")


class BorrowerInput(BaseModel):
    income: float
    credit_score: float
    loan_amount: float
    LTV: float
    dtir1: float


@app.post("/risk-score")
def get_risk_score(data: BorrowerInput):
    input_data = pd.DataFrame([data.dict()])
    full_input = pd.DataFrame(columns=model_columns)
    full_input.loc[0] = 0
    for col in input_data.columns:
        if col in full_input.columns:
            full_input[col] = input_data[col]
    prob = model.predict_proba(full_input)[0][1]
    expected_loss = prob * data.loan_amount
    if expected_loss < 50000:
        action = "Send automated reminder"
        risk = "Low"

    elif expected_loss < 200000:
        action = "Contact borrower"
        risk = "Medium"

    elif expected_loss < 500000:
        action = "Assign recovery officer"
        risk = "High"

    else:
        action = "Legal escalation"
        risk = "Critical"

    return {
        "default_probability": float(prob),
        "expected_loss": float(expected_loss),
        "risk_level": risk,
        "recommended_action": action
    }