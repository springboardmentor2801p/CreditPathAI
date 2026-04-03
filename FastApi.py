from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sqlite3

from recommendation_engine import recommendation_engine
from recommendation_engine_user import borrower_recommendation

app = FastAPI(title="CreditPath AI Risk Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("xgboost_model.pkl")


# -----------------------------
# USER FRIENDLY INPUT
# -----------------------------
class BorrowerInput(BaseModel):

    loan_amount: float
    monthly_income: float
    existing_emi: float
    property_value: float

    credit_score: float
    age: int

    loan_term: int
    occupancy_type: int
    business_loan: int


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/risk-score")
def get_risk_score(data: BorrowerInput):

    df = pd.DataFrame(columns=model.get_booster().feature_names)
    df.loc[0] = 0.0

    # # remove target
    # df = df.drop(columns=["Status"])

    # drop_cols = ["construction_type", "Secured_by"]
    # df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # leakage_features = [
    #     "Interest_rate_spread",
    #     "Upfront_charges",
    #     "rate_of_interest",
    #     "interest_burden"
    # ]

    # df = df.drop(columns=[c for c in leakage_features if c in df.columns])

    # if "year" in df.columns:
    #     df = df.drop(columns=["year"])


    # -------------------------
    # DERIVED FEATURES
    # -------------------------

    dtir1 = data.existing_emi / max(data.monthly_income, 1)
    loan_income_ratio = data.loan_amount / max(data.monthly_income, 1)
    ltv = data.loan_amount / max(data.property_value, 1)

    # -------------------------
    # MAP TO MODEL
    # -------------------------

    # -------------------------
    # MAP TO MODEL
    # -------------------------

    df.loc[0, "loan_amount"] = data.loan_amount
    df.loc[0, "income"] = data.monthly_income * 12
    df.loc[0, "Credit_Score"] = data.credit_score

    df.loc[0, "dtir1"] = dtir1
    df.loc[0, "loan_income_ratio"] = loan_income_ratio
    df.loc[0, "LTV"] = ltv

    df.loc[0, "age"] = data.age
    df.loc[0, "loan_term"] = data.loan_term
    df.loc[0, "occupancy_type"] = data.occupancy_type
    df.loc[0, "business_or_commercial"] = data.business_loan
    df = df[model.get_booster().feature_names]

    agent_result = recommendation_engine(model, df)
    user_result = borrower_recommendation(df,agent_result["default_probability"])

    return {
    "agent_recommendation": agent_result,
    "borrower_recommendation": user_result
    }
