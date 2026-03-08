# CreditPathAI

CreditPathAI is a machine learning–based system designed to predict borrower default risk and recommend appropriate loan recovery actions.

The system analyzes borrower financial data and loan attributes to estimate default probability, calculate expected loss, and assist recovery agents with actionable insights.

## Features
- Data preprocessing pipeline for borrower and loan datasets
- Machine learning models: Logistic Regression, XGBoost, LightGBM
- Risk prediction using the best-performing model (XGBoost)
- Expected loss calculation for prioritizing recovery
- Recommendation engine for recovery actions
- FastAPI backend for real-time predictions

## Example Output
Default Probability: 0.075  
Expected Loss: 1909.57  
Risk Level: Low  
Recommended Action: Send automated reminder

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost, LightGBM
- FastAPI
- SQLite

## Project Status
Backend system and API are completed.  
Interactive frontend dashboard (React + Plotly) is currently under development.
