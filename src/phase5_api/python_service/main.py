"""
CreditPathAI — Phase 5 Python ML Inference Service
FastAPI server on port 8000.
Express.js calls this service; never expose it directly to clients.
"""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("creditpathai.inference")

# ── Import inference engine ───────────────────────────────────────────────────
_THIS_DIR     = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.phase5_api.python_service.inference import MLInferenceEngine  # noqa

# ── Lifespan: load model once at startup ──────────────────────────────────────
engine: Optional[MLInferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Loading ML model...")
    engine = MLInferenceEngine()
    logger.info("ML model ready")
    yield
    logger.info("Shutting down inference service")


app = FastAPI(
    title="CreditPathAI Inference Service",
    version="1.0.0",
    description="Internal ML service — call via Express.js API only",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request / Response schemas (match production TSV columns exactly) ─────────
class BorrowerInput(BaseModel):
    residentialState:       str   = Field(..., example="CA")
    yearsEmployment:        str   = Field(..., example="2-5 years")
    homeOwnership:          str   = Field(..., example="rent")
    annualIncome:           float = Field(..., gt=0, example=60000)
    incomeVerified:         int   = Field(..., ge=0, le=1, example=1)
    dtiRatio:               float = Field(..., ge=0, example=18.5)
    lengthCreditHistory:    int   = Field(..., ge=0, example=5)
    numTotalCreditLines:    int   = Field(..., ge=0, example=15)
    numOpenCreditLines:     int   = Field(..., ge=0, example=10)
    numOpenCreditLines1Year:int   = Field(..., ge=0, example=6)
    revolvingBalance:       float = Field(..., ge=0, example=12000)
    revolvingUtilizationRate: float = Field(..., ge=0, le=150, example=65.0)
    numDerogatoryRec:       int   = Field(..., ge=0, example=0)
    numDelinquency2Years:   int   = Field(..., ge=0, example=0)
    numChargeoff1year:      int   = Field(..., ge=0, example=0)
    numInquiries6Mon:       int   = Field(..., ge=0, example=1)

    @field_validator("residentialState")
    @classmethod
    def state_uppercase(cls, v):
        return v.strip().upper()

    @field_validator("homeOwnership", "yearsEmployment")
    @classmethod
    def lower_strip(cls, v):
        return v.strip().lower()


class LoanInput(BaseModel):
    purpose:            str   = Field(..., example="debtconsolidation")
    isJointApplication: int   = Field(..., ge=0, le=1, example=0)
    loanAmount:         float = Field(..., gt=0, example=20000)
    term:               str   = Field(..., example="60 months")
    interestRate:       float = Field(..., gt=0, example=9.5)
    monthlyPayment:     float = Field(..., gt=0, example=420)
    grade:              str   = Field(..., example="C3")

    @field_validator("purpose", "term")
    @classmethod
    def lower_strip(cls, v):
        return v.strip().lower()

    @field_validator("grade")
    @classmethod
    def grade_upper(cls, v):
        return v.strip().upper()


class PredictRequest(BaseModel):
    borrower: BorrowerInput
    loan:     LoanInput


class PredictResponse(BaseModel):
    request_id:  str
    p_default:   float
    risk_score:  int
    risk_tier:   str
    confidence:  float
    latency_ms:  float


class RecommendResponse(BaseModel):
    request_id:           str
    decision:             str
    risk_tier:            str
    risk_score:           int
    p_default:            float
    interest_rate_min:    float
    interest_rate_max:    float
    interest_rate_rec:    float
    max_loan_amount:      float
    conditions:           list
    improvement_tips:     list
    explanation:          str
    latency_ms:           float


# ── Middleware: request logging ───────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid  = str(uuid.uuid4())[:8]
    t0   = time.perf_counter()
    request.state.request_id = rid
    logger.info(f"[{rid}] --> {request.method} {request.url.path}")
    resp = await call_next(request)
    ms   = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(f"[{rid}] <-- {resp.status_code}  {ms}ms")
    return resp


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    return {
        "status":     "healthy",
        "model":      type(engine.model).__name__ if engine else "not loaded",
        "features":   len(engine.feature_names)  if engine else 0,
        "service":    "CreditPathAI ML Inference",
        "version":    "1.0.0",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    if engine is None:
        raise HTTPException(503, detail="Model not loaded")

    rid = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    t0  = time.perf_counter()

    try:
        result = engine.predict(
            body.borrower.model_dump(),
            body.loan.model_dump(),
        )
    except Exception as exc:
        logger.error(f"[{rid}] Prediction error: {exc}")
        raise HTTPException(500, detail=f"Inference error: {str(exc)}")

    ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"[{rid}] tier={result['risk_tier']}  "
        f"score={result['risk_score']}  p={result['p_default']}  {ms}ms"
    )
    return {**result, "request_id": rid, "latency_ms": ms}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(body: PredictRequest, request: Request):
    if engine is None:
        raise HTTPException(503, detail="Model not loaded")

    rid = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    t0  = time.perf_counter()

    try:
        borrower_dict = body.borrower.model_dump()
        loan_dict     = body.loan.model_dump()
        prediction    = engine.predict(borrower_dict, loan_dict)
        rec           = engine.recommend(prediction, borrower_dict, loan_dict)
    except Exception as exc:
        logger.error(f"[{rid}] Recommendation error: {exc}")
        raise HTTPException(500, detail=f"Inference error: {str(exc)}")

    ms = round((time.perf_counter() - t0) * 1000, 2)
    return {**rec, "request_id": rid, "latency_ms": ms}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
