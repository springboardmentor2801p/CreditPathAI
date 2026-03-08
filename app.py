from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# load model
model = joblib.load("xgb_model.pkl")


def recommendation_engine(input_data):

    prob = model.predict_proba(input_data)[0][1]
    loan_amount = input_data["loanamount"].values[0]

    expected_loss = prob * loan_amount

    if expected_loss < 50000:
        risk_level = "Low"
        action = "Send automated reminder"

    elif expected_loss < 200000:
        risk_level = "Medium"
        action = "Call borrower and discuss repayment plan"

    elif expected_loss < 500000:
        risk_level = "High"
        action = "Assign recovery officer"

    else:
        risk_level = "Critical"
        action = "Escalate to recovery team"

    return {
    "loan_amount": float(loan_amount),
    "default_probability": float(prob),
    "expected_loss": float(expected_loss),
    "risk_level": risk_level,
    "recommended_action": action
}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    result = recommendation_engine(df)

    return result