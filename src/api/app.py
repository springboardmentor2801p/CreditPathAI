
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np


from fastapi.middleware.cors import CORSMiddleware
from src.api.input_builder import build_input
from src.recommendation.recommend import get_recommendation

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained pipeline
model = joblib.load("creditpath_pipeline.pkl")


@app.get("/")
def home():
    return {"message": "CreditPathAI API Running 🚀"}



@app.post("/predict")
def predict(data: dict):

    try:
        # ✅ Step 1: Build input
        full_input = build_input(data)

        # ✅ Step 2: Create DataFrame
        df = pd.DataFrame([full_input])

        # 🔥 Step 3: Match EXACT training columns
        if hasattr(model, "feature_names_in_"):
            df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # 🔥 Step 4: Minimal cleaning (DO NOT OVERDO)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(0)

        # ❌ NO astype(float)
        # ❌ NO apply(to_numeric)
        # ❌ NO manual conversions

        prob = model.predict_proba(df)[0][1]

        # 🔥 SAFE CASE OVERRIDE
        if (
            data.get("income", 0) > data.get("loan_amount", 0) and
            data.get("credit_score", 0) > 750 and
            data.get("ltv", 100) < 60 and
            data.get("dtir1", 100) < 20
        ):
            prob = min(prob, 0.15)

        # ✅ Step 6: Recommendation
        result = get_recommendation(prob, data)

        return result

    except Exception as e:
        return {"error": str(e)}

