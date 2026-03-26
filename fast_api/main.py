"""
================================================================================
  CreditPathAI — fast_api/main.py
  Purpose : FastAPI server that wraps the recommendation engine.
            Imports model loading and scoring directly from
            recommendation_engine/recommend_engine.py — no logic is duplicated.

  Run (from repo root):
      uvicorn fast_api.main:app --reload --host 0.0.0.0 --port 8000

  Interactive docs:
      http://localhost:8000/docs      (Swagger UI)
      http://localhost:8000/redoc     (ReDoc)
================================================================================
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure the repo root is on sys.path so the recommendation_engine package
#    can be imported correctly regardless of where uvicorn is launched from. ────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Import ONLY from the recommendation_engine package (no logic duplicated) ──
from recommendation_engine.recommend_engine import load_model, recommend  # noqa: E402
from fast_api.schemas import BorrowerRequest, RecommendationResponse, HealthResponse  # noqa: E402


# ─────────────────────────── App-level State ─────────────────────────────────

class _ModelState:
    """Holds the three artefacts loaded once at startup."""
    model         = None
    preprocessor  = None
    feature_names: list[str] = []


_state = _ModelState()


# ─────────────────────────── Lifespan (startup / shutdown) ───────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the XGBoost model and preprocessor when the server starts."""
    print("[CreditPathAI API] Loading recommendation engine model …")
    try:
        _state.model, _state.preprocessor, _state.feature_names = load_model()
        print(
            f"[CreditPathAI API] Model ready — "
            f"{len(_state.feature_names)} features loaded."
        )
    except FileNotFoundError as exc:
        # Log clearly; the /health endpoint will report model_loaded=False
        print(f"[CreditPathAI API] WARNING — model not found: {exc}")

    yield  # Server is running

    # Shutdown (nothing to clean up for joblib models)
    print("[CreditPathAI API] Shutting down.")


# ─────────────────────────── FastAPI Application ─────────────────────────────

app = FastAPI(
    title="CreditPathAI Recommendation API",
    description=(
        "REST API that accepts borrower financial features and returns a full "
        "credit-risk recommendation powered by an XGBoost default-prediction model.\n\n"
        "**Endpoint summary**\n"
        "- `POST /recommend` — score a single borrower\n"
        "- `GET  /health`    — check if the model is loaded and ready\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins during development; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Routes ──────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def health() -> HealthResponse:
    """
    Returns the current health status of the API.

    - **status**: `"ok"` if the model is loaded, `"degraded"` otherwise.
    - **model_loaded**: `true` when the XGBoost model is in memory.
    - **n_features**: number of input features the model expects.
    """
    loaded = _state.model is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        n_features=len(_state.feature_names),
    )


@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Score a borrower and return a credit-risk recommendation",
    tags=["Recommendation"],
)
def recommend_borrower(request: BorrowerRequest) -> RecommendationResponse:
    """
    **Accepts** a borrower's financial profile and **returns** a full
    risk-assessment and action recommendation.

    ### What happens internally
    1. The request body is converted to a plain `dict`.
    2. The `threshold` field is extracted (default 0.50) and removed from the
       feature dict so it is not passed to the model.
    3. `recommend()` from `recommendation_engine.recommend_engine` is called —
       no model logic lives in this file.
    4. The result dict is returned as a structured JSON response.

    ### Response highlights
    | Field | Description |
    |---|---|
    | `default_probability` | Model P(default), 0–1 |
    | `predicted_default` | `true` if probability ≥ threshold |
    | `risk_band` | Very Low / Low / Medium / High / Very High |
    | `expected_loss` | P(default) × loan amount |
    | `priority_level` | Low / Medium / High / Critical |
    | `recommended_action` | Primary next step |
    | `risk_flags` | List of detected red-flags |
    """
    if _state.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Ensure that the XGBoost artefacts exist at "
                "training/advanced/saved_models/ and restart the server."
            ),
        )

    # Convert Pydantic model → plain dict; extract optional threshold override
    borrower_dict: dict = request.model_dump()
    threshold: float    = borrower_dict.pop("threshold", 0.50)

    try:
        result: dict = recommend(
            borrower       = borrower_dict,
            model          = _state.model,
            preprocessor   = _state.preprocessor,
            feature_names  = _state.feature_names,
            threshold      = threshold,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Missing feature required by the model: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation engine error: {exc}",
        ) from exc

    return RecommendationResponse(**result)


@app.get(
    "/random-borrower",
    summary="Get a random real borrower from the dataset",
    tags=["Utility"],
)
def get_random_borrower():
    """
    Fetches a random borrower from the underlying processed_loans database
    to pre-fill frontend UI testing forms with real training distributions.
    """
    import sqlite3
    import pandas as pd
    import numpy as np
    
    db_path = os.path.join(_REPO_ROOT, "csv2database", "creditpathai.db")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
        
    try:
        conn = sqlite3.connect(db_path)
        # Fetch 1 random row
        df = pd.read_sql("SELECT * FROM processed_loans ORDER BY RANDOM() LIMIT 1", conn)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No loans found in database")
            
        row_dict = df.iloc[0].to_dict()
        
        # Clean up numpy types for JSON serialization
        for k, v in row_dict.items():
            if isinstance(v, (np.int64, np.int32)):
                row_dict[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                row_dict[k] = float(v)
                
        return row_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/batch-cases",
    summary="Get a batch of scored borrowers for dashboard visualization",
    tags=["Utility"],
)
def get_batch_cases(n: int = 15):
    """
    Fetches N random borrowers from the database and scores them using the recommendation engine.
    Used to populate the frontend dashboards with real portfolio data.
    """
    import sqlite3
    import pandas as pd
    import numpy as np
    
    if _state.model is None or _state.preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    db_path = os.path.join(_REPO_ROOT, "csv2database", "creditpathai.db")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
        
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM processed_loans ORDER BY RANDOM() LIMIT {n}", conn)
        conn.close()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No loans found in database")
        
        results = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Clean up numpy types
            for k, v in row_dict.items():
                if isinstance(v, (np.int64, np.int32)):
                    row_dict[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    row_dict[k] = float(v)
            
            # Score
            try:
                rec = recommend(
                    borrower=row_dict,
                    model=_state.model,
                    preprocessor=_state.preprocessor,
                    feature_names=_state.feature_names,
                    threshold=0.50
                )
                
                # Add borrower ID
                rec['id'] = f"BRW-{np.random.randint(1000, 9999)}"
                rec['loan_amount'] = row_dict.get('loanAmount', 0)
                results.append(rec)
            except Exception as e:
                continue
                
        return {"cases": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


