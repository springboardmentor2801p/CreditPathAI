import pandas as pd
import sqlite3
import joblib
from recommendation_engine import recommendation_engine

model = joblib.load("src/xgboost_model.pkl")

conn = sqlite3.connect("src/creditpathai.db")

df = pd.read_sql("SELECT * FROM processed_loans LIMIT 1", conn)
# df = pd.read_sql("SELECT * FROM processed_loans ORDER BY RANDOM() LIMIT 1", conn)

conn.close()

# Remove target
df = df.drop(columns=["Status"])

# Apply SAME drops used in training
drop_cols = ["construction_type", "Secured_by"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

leakage_features = [
    "Interest_rate_spread",
    "Upfront_charges",
    "rate_of_interest",
    "interest_burden"
]

df = df.drop(columns=[col for col in leakage_features if col in df.columns])

if "year" in df.columns:
    df = df.drop(columns=["year"])

result = recommendation_engine(model, df)

print("\nFormatted Output\n")

print(f"Default Probability: {result['default_probability']}")
print(f"Expected Loss: {result['expected_loss']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommended Action: {result['recommended_action']}") 

# Show the row used for prediction
print("\nBorrower Row Used For Prediction\n")
print(df)

# Get probability manually from model
prob = model.predict_proba(df)[0][1]

# Get loan amount
loan_amount = df["loan_amount"].values[0]

print("\nVerification Values\n")
print("Loan Amount:", loan_amount)
print("Model Probability:", prob)