from fastapi import FastAPI
from pydantic import BaseModel
from recommendation_engine import risk_scoring_engine, prepare_input, model

app = FastAPI(title="CreditPath AI Risk Scoring API")

# input structure
class BorrowerInput(BaseModel):
    purpose: str
    isJointApplication: float
    loanAmount: float
    term: str
    interestRate: float
    monthlyPayment: int
    grade: str
    residentialState: str
    yearsEmployment: str
    homeOwnership: str
    annualIncome: int
    incomeVerified: int
    dtiRatio: float
    lengthCreditHistory: int
    numTotalCreditLines: int
    numOpenCreditLines: float
    numOpenCreditLines1Year: int
    revolvingBalance: int
    revolvingUtilizationRate: float
    numDerogatoryRec: int
    numDelinquency2Years: int
    numChargeoff1year: int
    numInquiries6Mon: int
   

# API endpoint
@app.post("/risk-score")
def get_risk_score(data: BorrowerInput):
    input_data = data.dict()
    input_df = prepare_input(input_data)
    result = risk_scoring_engine(input_df, model)
    return result

inp = BorrowerInput(
    purpose="debtconsolidation",
    isJointApplication=0,
    loanAmount=40000000,
    term="36 months",
    interestRate=15,
    monthlyPayment=13500,
    grade="E3",
    residentialState="CA",
    yearsEmployment="10+ years",
    homeOwnership="mortgage",
    annualIncome=20000,
    incomeVerified=0,
    dtiRatio=18.4,
    lengthCreditHistory=12,
    numTotalCreditLines=2,
    numOpenCreditLines=1,
    numOpenCreditLines1Year=1,
    revolvingBalance=7500,
    revolvingUtilizationRate=4.25,
    numDerogatoryRec=0,
    numDelinquency2Years=0,
    numChargeoff1year=0,
    numInquiries6Mon=0
)

# Run FastAPI Server:
# uvicorn app:app –reload
# You will see:
# Uvicorn running on http://127.0.0.1:8000