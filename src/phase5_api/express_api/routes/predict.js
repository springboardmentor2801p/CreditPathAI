const express = require("express");
const axios   = require("axios");
const router  = express.Router();
const { validatePredictRequest } = require("../middleware/validate");

const PYTHON_URL = process.env.PYTHON_SERVICE_URL || "http://localhost:8000";
const TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || "30000");

// ── Helper: forward to Python service ────────────────────────────────────────
async function callPython(endpoint, payload) {
  const response = await axios.post(`${PYTHON_URL}${endpoint}`, payload, {
    timeout: TIMEOUT_MS,
    headers: { "Content-Type": "application/json" },
  });
  return response.data;
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /health-check
// Returns status of both Express and Python services
// ─────────────────────────────────────────────────────────────────────────────
router.get("/health-check", async (req, res) => {
  const expressStatus = { status: "healthy", service: "Express API", version: "1.0.0" };

  try {
    const pyStatus = await axios.get(`${PYTHON_URL}/health`, { timeout: 5000 });
    return res.json({
      success: true,
      express: expressStatus,
      python:  pyStatus.data,
    });
  } catch (err) {
    return res.status(503).json({
      success: false,
      express: expressStatus,
      python:  { status: "unreachable", error: err.message },
    });
  }
});


// ─────────────────────────────────────────────────────────────────────────────
// POST /predict-risk
// Input:  { borrower: {...}, loan: {...} }
// Output: { p_default, risk_score, risk_tier, confidence, latency_ms }
// ─────────────────────────────────────────────────────────────────────────────
router.post("/predict-risk", validatePredictRequest, async (req, res) => {
  const { borrower, loan } = req.body;

  try {
    const result = await callPython("/predict", { borrower, loan });
    return res.json({
      success:    true,
      request_id: result.request_id,
      data: {
        p_default:  result.p_default,
        risk_score: result.risk_score,
        risk_tier:  result.risk_tier,
        confidence: result.confidence,
      },
      meta: { latency_ms: result.latency_ms },
    });
  } catch (err) {
    return handleAxiosError(err, res, "predict-risk");
  }
});


// ─────────────────────────────────────────────────────────────────────────────
// POST /get-recommendation
// Input:  { borrower: {...}, loan: {...} }
// Output: full recommendation including decision, rates, conditions, tips
// ─────────────────────────────────────────────────────────────────────────────
router.post("/get-recommendation", validatePredictRequest, async (req, res) => {
  const { borrower, loan } = req.body;

  try {
    const result = await callPython("/recommend", { borrower, loan });
    return res.json({
      success:    true,
      request_id: result.request_id,
      data: {
        decision:          result.decision,
        risk_tier:         result.risk_tier,
        risk_score:        result.risk_score,
        p_default:         result.p_default,
        interest_rates: {
          minimum:     result.interest_rate_min,
          maximum:     result.interest_rate_max,
          recommended: result.interest_rate_rec,
        },
        max_loan_amount:   result.max_loan_amount,
        conditions:        result.conditions,
        improvement_tips:  result.improvement_tips,
        explanation:       result.explanation,
      },
      meta: { latency_ms: result.latency_ms },
    });
  } catch (err) {
    return handleAxiosError(err, res, "get-recommendation");
  }
});


// ── Error handler for Axios upstream calls ────────────────────────────────────
function handleAxiosError(err, res, route) {
  if (err.response) {
    // Python service returned an error
    return res.status(err.response.status).json({
      success: false,
      error:   "ML service error",
      detail:  err.response.data?.detail || err.message,
    });
  }
  if (err.code === "ECONNREFUSED" || err.code === "ECONNABORTED") {
    return res.status(503).json({
      success: false,
      error:   "ML inference service unavailable",
      detail:  "Python service is not running. Start it with: uvicorn main:app --port 8000",
    });
  }
  if (err.code === "ETIMEDOUT") {
    return res.status(504).json({
      success: false,
      error:   "ML inference timeout",
      detail:  `Request to /${route} timed out after ${TIMEOUT_MS}ms`,
    });
  }
  return res.status(500).json({
    success: false,
    error:   "Internal server error",
    detail:  err.message,
  });
}

module.exports = router;
