from fastapi import FastAPI
import pickle
from recommendation_engine import generate_recommendation

app = FastAPI()

# Load trained model
model = pickle.load(open("final_model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Loan Risk Scoring API Running"}

@app.post("/predict")
def predict(probability: float, loan_amount: float):

    result = generate_recommendation(probability, loan_amount)

    return result
