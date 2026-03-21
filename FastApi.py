from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from recommendation_engine import recommendation_engine

# create API
app = FastAPI(title="CreditPath AI Risk Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load trained model
model = joblib.load("xgboost_model.pkl")


# -----------------------------
# INPUT STRUCTURE
# -----------------------------
class BorrowerInput(BaseModel):
    loan_amount: float
    income: float
    credit_score: float
    debt_ratio: float
    missed_payments: int


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.post("/risk-score")
def get_risk_score(data: BorrowerInput):

    import sqlite3

    # load a sample borrower row
    conn = sqlite3.connect("creditpathai.db")
    df = pd.read_sql("SELECT * FROM processed_loans LIMIT 1", conn)
    conn.close()

    # remove training-only columns
    df = df.drop(columns=["Status"])

    drop_cols = ["construction_type", "Secured_by"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    leakage_features = [
        "Interest_rate_spread",
        "Upfront_charges",
        "rate_of_interest",
        "interest_burden"
    ]

    df = df.drop(columns=[c for c in leakage_features if c in df.columns])

    if "year" in df.columns:
        df = df.drop(columns=["year"])

    # replace fields using API input
    df["loan_amount"] = data.loan_amount
    df["income"] = data.income

    # run recommendation engine
    result = recommendation_engine(model, df)

    return result