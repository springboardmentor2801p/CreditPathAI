from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

from fastapi.middleware.cors import CORSMiddleware

# ✅ Create FastAPI app
app = FastAPI()

# ✅ Enable CORS (for React later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load trained model
model = pickle.load(open("lightgbm_model.pkl", "rb"))

# ✅ Input schema (MATCH MODEL FEATURES)
class BorrowerInput(BaseModel):
    person_age: float
    person_income: float
    person_emp_length: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: int
    cb_person_cred_hist_length: float

# ✅ API endpoint
@app.post("/risk-score")
def predict(data: BorrowerInput):

    try:
        # ✅ Step 1: Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # ✅ Step 2: Add missing columns
        for col in model.feature_name_:
            if col not in df.columns:
                df[col] = 0

        # ✅ Step 3: Arrange columns in correct order
        df = df[model.feature_name_]

        # ✅ Step 4: Prediction
        prob = model.predict_proba(df)[0][1]

        # ✅ Step 5: Business logic
        expected_loss = prob * data.loan_amnt

        if expected_loss < 50000:
            risk = "Low"
            action = "Send reminder"
        elif expected_loss < 200000:
            risk = "Medium"
            action = "Contact borrower"
        elif expected_loss < 500000:
            risk = "High"
            action = "Assign recovery officer"
        else:
            risk = "Critical"
            action = "Legal action"

        # ✅ Step 6: Return response
        return {
            "default_probability": round(prob, 4),
            "expected_loss": round(expected_loss, 2),
            "risk_level": risk,
            "recommended_action": action
        }

    except Exception as e:
        return {"error": str(e)}