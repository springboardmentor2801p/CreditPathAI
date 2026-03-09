require("dotenv").config();

const express = require("express");
const helmet  = require("helmet");
const morgan  = require("morgan");

const predictRouter = require("./routes/predict");

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Security + parsing ────────────────────────────────────────────────────────
app.use(helmet());
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));

// ── HTTP request logging ──────────────────────────────────────────────────────
app.use(
  morgan(":method :url :status :res[content-length] - :response-time ms")
);

// ── Routes ────────────────────────────────────────────────────────────────────
app.use("/api", predictRouter);

// ── Root ──────────────────────────────────────────────────────────────────────
app.get("/", (req, res) => {
  res.json({
    name:      "CreditPathAI API",
    version:   "1.0.0",
    phase:     "Phase 5 — Backend API",
    endpoints: [
      "GET  /api/health-check",
      "POST /api/predict-risk",
      "POST /api/get-recommendation",
    ],
    docs: "See API_DOCS.md for request/response schemas",
  });
});

// ── 404 handler ───────────────────────────────────────────────────────────────
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error:   `Route ${req.method} ${req.url} not found`,
  });
});

// ── Global error handler ──────────────────────────────────────────────────────
app.use((err, req, res, next) => {  // eslint-disable-line no-unused-vars
  console.error("[UNHANDLED]", err.stack);
  res.status(500).json({
    success: false,
    error:   "Unexpected server error",
    detail:  process.env.NODE_ENV === "development" ? err.message : undefined,
  });
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`CreditPathAI Express API running on http://localhost:${PORT}`);
  console.log(`Python ML service expected at: ${process.env.PYTHON_SERVICE_URL}`);
});

module.exports = app;
