
# ===============================
# FASTAPI MAIN
# CreditPath AI
# ===============================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import prepare_input, predict_default, generate_recommendation
import joblib

# Load model and scaler
model, scaler = joblib.load("creditpath_model.pkl"), joblib.load("creditpath_scaler.pkl")

app = FastAPI()

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data structure
class BorrowerInput(BaseModel):
    Credit_Score: float
    loan_amount: float
    income: float
    LTV: float
    dtir1: float

@app.get("/")
def root():
    return {"message": "CreditPath AI Backend Running!"}

@app.post("/predict")
def predict(data: BorrowerInput):
    # Prepare input using predict.py
    input_df = prepare_input(
        credit_score=data.Credit_Score,
        loan_amount=data.loan_amount,
        income=data.income,
        ltv=data.LTV,
        dtir1=data.dtir1
    )

    # Predict using predict.py
    prob = predict_default(model, scaler, input_df)

    # Get full recommendation using predict.py
    result = generate_recommendation(prob, data.loan_amount)

    return result