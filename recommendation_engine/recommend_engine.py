"""
================================================================================
  CreditPathAI — recommendation_engine/recommend_engine.py
  Purpose : Load the XGBoost model, score a borrower, and display the full
            risk assessment and recommended action.

  USAGE
  ─────
  # Demo mode (synthetic high-risk borrower):
  python recommendation_engine/recommend_engine.py --demo

  # Score from a CSV file:
  python recommendation_engine/recommend_engine.py --input path/to/loans.csv

  # Score from the SQLite database (default: 500 rows):
  python recommendation_engine/recommend_engine.py --batch

  # Change decision threshold:
  python recommendation_engine/recommend_engine.py --demo --threshold 0.35

  # Save batch results to CSV:
  python recommendation_engine/recommend_engine.py --batch --output results.csv
================================================================================
"""

from __future__ import annotations

import sys, io, os, json, argparse, sqlite3

# ── UTF-8 stdout (safe on Windows) ───────────────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

# ── Resolve repo root so the script can be run from anywhere ─────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from recommendation_engine.risk_rules import (
    resolve_priority,
    get_action_plan,
    prob_to_risk_band,
    detect_risk_flags,
)

# ─────────────────────────── Model Paths ─────────────────────────────────────
_SAVED_MODELS     = os.path.join(_REPO_ROOT, "training", "advanced", "saved_models")
MODEL_PATH        = os.path.join(_SAVED_MODELS, "xgboost.joblib")
PREPROCESSOR_PATH = os.path.join(_SAVED_MODELS, "preprocessor.joblib")
METADATA_PATH     = os.path.join(_SAVED_MODELS, "metadata.json")

DEFAULT_THRESHOLD = 0.50   # P(default) >= this → predicted default


# ─────────────────────────── Feature Engineering ─────────────────────────────

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 5 interaction features used during model training."""
    df = df.copy()
    df["interest_dti_burden"] = df["interestRate"] * df["dtiRatio"]
    df["credit_stress_index"] = (
        (df["numDelinquency2Years"] + df["numDerogatoryRec"])
        / (df["lengthCreditHistory"] + 1)
    )
    df["payment_pressure"] = df["monthlyPayment"] / ((df["annualIncome"] / 12) + 1e-6)
    df["credit_depth"]     = np.log1p(df["numTotalCreditLines"])
    df["revolving_load"]   = df["revolvingBalance"] / (df["annualIncome"] + 1)
    return df


def _prepare(input_data: dict | pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Convert borrower dict or DataFrame into a model-ready DataFrame."""
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data.copy()
    if "loanStatus" in df.columns:
        df = df.drop(columns=["loanStatus"])
    df = _add_interaction_features(df)
    return df[feature_names]


# ─────────────────────────── Model Loading ───────────────────────────────────

def load_model() -> tuple:
    """
    Load and return (model, preprocessor, feature_names).
    Raises FileNotFoundError if any artefact is missing.
    """
    for label, path in [
        ("Model",        MODEL_PATH),
        ("Preprocessor", PREPROCESSOR_PATH),
        ("Metadata",     METADATA_PATH),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found at:\n  {path}")

    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    print(
        f"[recommend_engine] XGBoost model loaded "
        f"({metadata.get('n_features', '?')} features, "
        f"trained {metadata.get('trained_at', 'unknown')})."
    )
    return model, preprocessor, feature_names


# ─────────────────────────── Core Scoring ────────────────────────────────────

def recommend(
    borrower: dict,
    model,
    preprocessor,
    feature_names: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Score a single borrower and return a full recommendation dict.

    Parameters
    ----------
    borrower      : dict of raw borrower features
    model         : fitted XGBoost classifier (joblib-loaded)
    preprocessor  : fitted ColumnTransformer (joblib-loaded)
    feature_names : ordered feature list from metadata.json
    threshold     : P(default) cutoff for binary predicted_default flag

    Returns
    -------
    {
      "default_probability"  : float  — model P(default)
      "predicted_default"    : bool   — True if prob >= threshold
      "risk_band"            : str    — Very Low / Low / Medium / High / Very High
      "loan_amount"          : float  — exposure (₹)
      "expected_loss"        : float  — probability × loan amount (₹)
      "priority_level"       : str    — Low / Medium / High / Critical
      "assigned_team"        : str
      "recovery_channel"     : str
      "follow_up_frequency"  : str
      "legal_action"         : bool
      "recommended_action"   : str
      "escalation_notes"     : str
      "risk_flags"           : list[str]  — qualitative red flags
    }
    """
    # ── Predict ──────────────────────────────────────────────────────────────
    X = _prepare(borrower, feature_names)
    X_t = preprocessor.transform(X)
    prob = float(model.predict_proba(X_t)[0][1])

    # ── Financial Exposure ────────────────────────────────────────────────────
    loan_amount   = float(borrower.get("loanAmount", 0))
    expected_loss = prob * loan_amount

    # ── Risk Classification (from risk_rules.py) ──────────────────────────────
    priority    = resolve_priority(expected_loss)
    action_plan = get_action_plan(priority)
    risk_band   = prob_to_risk_band(prob)
    risk_flags  = detect_risk_flags(borrower)

    return {
        # ML scores
        "default_probability": round(prob, 6),
        "predicted_default":   prob >= threshold,
        "risk_band":           risk_band,
        # Financial exposure
        "loan_amount":         loan_amount,
        "expected_loss":       round(expected_loss, 2),
        # Action plan (all fields from risk_rules ACTION_PLAYBOOKS)
        "priority_level":      action_plan["priority"],
        "assigned_team":       action_plan["assigned_team"],
        "recovery_channel":    action_plan["recovery_channel"],
        "follow_up_frequency": action_plan["follow_up_frequency"],
        "legal_action":        action_plan["legal_action"],
        "recommended_action":  action_plan["recommended_action"],
        "escalation_notes":    action_plan["escalation_notes"],
        # Qualitative flags
        "risk_flags":          risk_flags,
    }


def recommend_batch(
    df: pd.DataFrame,
    model,
    preprocessor,
    feature_names: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame and return an enriched results DataFrame,
    sorted by expected_loss descending (highest risk first).
    """
    results = []
    for _, row in df.iterrows():
        rec = recommend(row.to_dict(), model, preprocessor, feature_names, threshold)
        results.append(rec)

    rec_df = pd.DataFrame(results, index=df.index)
    # Flatten risk_flags list → pipe-separated string for CSV friendliness
    rec_df["risk_flags"] = rec_df["risk_flags"].apply(
        lambda flags: " | ".join(flags) if flags else "None"
    )

    result = pd.concat([df.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)
    return result.sort_values("expected_loss", ascending=False)


# ─────────────────────────── Display Helpers ─────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 70}\n  {title}\n{'─' * 70}")


def display_recommendation(rec: dict) -> None:
    """Pretty-print a single borrower recommendation to the console."""
    _section("Risk Assessment & Recommendation")
    w = 26

    def row(k, v):
        print(f"  {k:<{w}}: {v}")

    print("  ── ML Score ──")
    row("Default Probability",  f"{rec['default_probability'] * 100:.2f}%")
    row("Predicted Default",    "YES ⚠" if rec["predicted_default"] else "NO ✓")
    row("Risk Band",            rec["risk_band"])

    print("\n  ── Financial Exposure ──")
    row("Loan Amount (₹)",      f"{rec['loan_amount']:,.0f}")
    row("Expected Loss (₹)",    f"{rec['expected_loss']:,.2f}")

    print("\n  ── Resolution Plan ──")
    row("Priority Level",       rec["priority_level"])
    row("Assigned Team",        rec["assigned_team"])
    row("Recovery Channel",     rec["recovery_channel"])
    row("Follow-up Frequency",  rec["follow_up_frequency"])
    row("Legal Action",         "YES ⚖" if rec["legal_action"] else "No")
    row("Recommended Action",   rec["recommended_action"])
    row("Escalation Notes",     rec["escalation_notes"])

    print("\n  ── Risk Flags ──")
    if rec["risk_flags"]:
        for flag in rec["risk_flags"]:
            print(f"    • {flag}")
    else:
        print("    None detected")


def display_batch_summary(result: pd.DataFrame) -> None:
    """Print a portfolio-level summary from a batch result DataFrame."""
    _section("Portfolio Summary")
    total         = len(result)
    n_defaults    = int(result["predicted_default"].sum())
    total_el      = result["expected_loss"].sum()
    avg_prob      = result["default_probability"].mean()
    n_legal       = int(result["legal_action"].sum())

    print(f"  Total borrowers        : {total:,}")
    print(f"  Predicted defaults     : {n_defaults:,} ({n_defaults / total * 100:.1f}%)")
    print(f"  Avg default probability: {avg_prob * 100:.2f}%")
    print(f"  Total expected loss (₹): {total_el:,.0f}")
    print(f"  Borrowers needing legal: {n_legal:,}")

    print("\n  Priority Distribution:")
    for level in ["Low", "Medium", "High", "Critical"]:
        count = int((result["priority_level"] == level).sum())
        pct   = count / total * 100
        el    = result.loc[result["priority_level"] == level, "expected_loss"].sum()
        bar   = "█" * int(pct / 2)
        print(f"    {level:<10} {count:>7,}  ({pct:5.1f}%)  EL: ₹{el:>14,.0f}  {bar}")

    print("\n  Risk Band Distribution:")
    for band in ["Very Low", "Low", "Medium", "High", "Very High"]:
        count = int((result["risk_band"] == band).sum())
        pct   = count / total * 100
        bar   = "█" * int(pct / 2)
        print(f"    {band:<10} {count:>7,}  ({pct:5.1f}%)  {bar}")


# ─────────────────────────── Demo Borrower ───────────────────────────────────

def _demo_borrower() -> dict:
    """Return a synthetic high-risk borrower with all required features."""
    return {
        "isJointApplication":           0,
        "loanAmount":                   350_000,
        "interestRate":                 19.5,
        "monthlyPayment":               9_800,
        "yearsEmployment":              2,
        "annualIncome":                 480_000,
        "dtiRatio":                     0.45,
        "lengthCreditHistory":          5,
        "numTotalCreditLines":          8,
        "numOpenCreditLines":           5,
        "numOpenCreditLines1Year":      2,
        "revolvingBalance":             120_000,
        "revolvingUtilizationRate":     0.82,
        "numDerogatoryRec":             1,
        "numDelinquency2Years":         3,
        "numChargeoff1year":            1,
        "numInquiries6Mon":             4,
        "term_months":                  36,
        "grade_score":                  5,
        "loan_to_income_ratio":         0.73,
        "payment_to_income_ratio":      0.245,
        "repayment_velocity":           0.028,
        "loan_amortization_rate":       0.033,
        "open_credit_ratio":            0.625,
        "recent_credit_velocity":       2,
        "inquiry_intensity":            0.67,
        "delinquency_density":          0.6,
        "derogatory_density":           0.2,
        "estimated_credit_limit":       146_000,
        "credit_utilization_recomputed":0.82,
        "log_loanAmount":               np.log1p(350_000),
        "log_annualIncome":             np.log1p(480_000),
        "log_revolvingBalance":         np.log1p(120_000),
        "incomeVerified":               1,
        "purpose_business":             0,
        "purpose_debtconsolidation":    1,
        "purpose_education":            0,
        "purpose_healthcare":           0,
        "purpose_homeimprovement":      0,
        "purpose_other":                0,
        "homeOwnership_own":            0,
        "homeOwnership_rent":           1,
    }

def _demo_borrower2() -> dict:
    """Return a synthetic high-risk borrower with all required features."""
    return {
        "isJointApplication":           0,
        "loanAmount":                   25_079,
        "interestRate":                 6.83,
        "monthlyPayment":               772,
        "yearsEmployment":              1,
        "annualIncome":                 50_204,
        "dtiRatio":                     0.45,
        "lengthCreditHistory":          5,
        "numTotalCreditLines":          7,
        "numOpenCreditLines":           6,
        "numOpenCreditLines1Year":      3,
        "revolvingBalance":             12_960,
        "revolvingUtilizationRate":     0.70,
        "numDerogatoryRec":             0,
        "numDelinquency2Years":         0,
        "numChargeoff1year":            0,
        "numInquiries6Mon":             0,
        "term_months":                  36,
        "grade_score":                  4.2,
        "loan_to_income_ratio":         0.49,
        "payment_to_income_ratio":      0.015,
        "repayment_velocity":           0.028,
        "loan_amortization_rate":       0.677,
        "open_credit_ratio":            0.75,
        "recent_credit_velocity":       0.42,
        "inquiry_intensity":            0,
        "delinquency_density":          0,
        "derogatory_density":           0,
        "estimated_credit_limit":       18_294,
        "credit_utilization_recomputed":0.708,
        "log_loanAmount":               np.log1p(25_079),
        "log_annualIncome":             np.log1p(50_204),
        "log_revolvingBalance":         np.log1p(120_000),
        "incomeVerified":               1,
        "purpose_business":             0,
        "purpose_debtconsolidation":    1,
        "purpose_education":            0,
        "purpose_healthcare":           0,
        "purpose_homeimprovement":      0,
        "purpose_other":                0,
        "homeOwnership_own":            0,
        "homeOwnership_rent":           1,
    }

# ─────────────────────────── CLI Entry Point ─────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CreditPathAI — Recommendation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo",      action="store_true",
                        help="Score a synthetic demo borrower (default if no flag given).")
    parser.add_argument("--demoTwo",      action="store_true",
                        help="Score a synthetic demo borrower (2nd demo).")
    parser.add_argument("--batch",     action="store_true",
                        help="Score all loans in the SQLite database.")
    parser.add_argument("--input",     default=None,
                        help="Path to a CSV file to score.")
    parser.add_argument("--output",    default=None,
                        help="Path to save batch results CSV.")
    parser.add_argument("--limit",     type=int, default=500,
                        help="Max rows for --batch DB mode.  [default: 500]")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold P(default).  [default: {DEFAULT_THRESHOLD}]")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    _section("Loading Model")
    model, preprocessor, feature_names = load_model()

    # ── Demo mode (single borrower) ───────────────────────────────────────────
    if args.demo or (not args.batch and not args.input):
        _section("Demo — Single Borrower Scoring")
        borrower = _demo_borrower()
        print("  Borrower features:")
        for k, v in borrower.items():
            print(f"    {k:<35}: {v}")

        rec = recommend(borrower, model, preprocessor, feature_names, args.threshold)
        display_recommendation(rec)
        return
    if args.demoTwo or (not args.batch and not args.input):
        _section("DemoTwo — Single Borrower Scoring")
        borrower = _demo_borrower2()
        print("  Borrower features:")
        for k, v in borrower.items():
            print(f"    {k:<35}: {v}")

        rec = recommend(borrower, model, preprocessor, feature_names, args.threshold)
        display_recommendation(rec)
        return

    # ── Batch / CSV mode ──────────────────────────────────────────────────────
    _section("Batch Scoring")

    if args.input:
        if not os.path.exists(args.input):
            print(f"  [ERROR] File not found: {args.input}")
            sys.exit(1)
        df = pd.read_csv(args.input)
        print(f"  Loaded {len(df):,} rows from {args.input}")
    else:
        db_path = os.path.join(_REPO_ROOT, "csv2database", "creditpathai.db")
        if not os.path.exists(db_path):
            print(f"  [ERROR] Database not found: {db_path}")
            sys.exit(1)
        conn = sqlite3.connect(db_path)
        df   = pd.read_sql(f"SELECT * FROM processed_loans LIMIT {args.limit}", conn)
        conn.close()
        print(f"  Loaded {len(df):,} rows from SQLite (LIMIT {args.limit})")

    result = recommend_batch(df, model, preprocessor, feature_names, args.threshold)
    display_batch_summary(result)

    # Top 5 highest-risk borrowers
    _section("Top 5 Highest Expected-Loss Borrowers")
    cols = ["loanAmount", "default_probability", "expected_loss",
            "priority_level", "assigned_team", "recommended_action"]
    available = [c for c in cols if c in result.columns]
    print(result.head(5)[available].to_string(index=True))

    # Save output
    out_path = args.output or os.path.join(_HERE, "batch_recommendations.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\n  ✓ Results saved → {out_path}")


if __name__ == "__main__":
    main()
