"""
================================================================================
  CreditPathAI — predict.py
  Purpose : Load a saved model and score loan default risk.
            Supports both the baseline (Logistic Regression) and
            advanced (XGBoost / LightGBM) model tiers.

USAGE EXAMPLES
--------------
  # List all saved models in both tiers
  python predict.py --list

  # Predict using best model from advanced tier (default)
  python predict.py

  # Choose tier + model explicitly
  python predict.py --tier baseline
  python predict.py --tier advanced --model xgboost
  python predict.py --tier advanced --model lightgbm

  # Predict on a CSV instead of the DB
  python predict.py --input new_loans.csv --tier advanced --model lightgbm

  # Lower threshold → catch more defaults (higher recall)
  python predict.py --threshold 0.35

  # Limit rows from DB (quick test)
  python predict.py --limit 500

ARGUMENTS
---------
  --tier       {baseline, advanced}          [default: advanced]
  --model      model key within the tier     [default: best model in tier]
  --input      CSV path                      [default: DB processed_loans]
  --output     save path for scored CSV      [default: predictions/<tier>_<model>.csv]
  --threshold  decision threshold 0–1        [default: 0.50]
  --list       print model info and exit
  --limit      max DB rows (0 = all)         [default: 0]
================================================================================
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import feature_pipeline as fp

# ─────────────────────────── Paths ───────────────────────────────────────────
TIER_DIRS = {
    "baseline" : os.path.join(BASE_DIR, "baseline", "saved_models"),
    "advanced" : os.path.join(BASE_DIR, "advanced", "saved_models"),
}
PRED_DIR  = os.path.join(BASE_DIR, "predictions")
DB_PATH   = os.path.join(BASE_DIR, "creditpathai.db")


# ─────────────────────────── Helpers ─────────────────────────────────────────
def section(t):
    print(f"\n{'─'*65}\n  {t}\n{'─'*65}")


def load_metadata(tier: str) -> dict:
    path = os.path.join(TIER_DIRS[tier], "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No metadata found for tier '{tier}' at {path}.\n"
            f"  → Run  python {tier}/train_{tier}.py  first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_all_models():
    """Print a formatted summary of every saved tier + model."""
    print("\n  CreditPathAI — Saved Models\n")
    for tier in ["baseline", "advanced"]:
        try:
            meta = load_metadata(tier)
        except FileNotFoundError as e:
            print(f"  [{tier.upper()}]  ← not trained yet\n    {e}\n")
            continue

        print(f"  ╔══ {tier.upper()} ({'Logistic Regression' if tier=='baseline' else 'XGBoost + LightGBM'}) ══")
        print(f"  ║  Trained at  : {meta['trained_at']}")
        print(f"  ║  Train rows  : {meta['train_rows']:,}  |  Test rows : {meta['test_rows']:,}")
        print(f"  ║  Features    : {meta['n_features']}")

        if tier == "baseline":
            m = meta["metrics"]
            print(f"  ║  {'Model':<20} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1(def)':>8} {'CV AUC':>8}")
            print(f"  ║  {'logistic_regression':<20} {m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f} "
                  f"{m['f1_default']:>8.4f} {m['cv_auc_mean']:>8.4f}")
        else:
            best = meta.get("best_model", "")
            print(f"  ║  {'Model':<20} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1(def)':>8} {'CV AUC':>8}")
            for key, info in meta["models"].items():
                m   = info["metrics"]
                tag = "  ← best" if key == best else ""
                print(f"  ║  {key:<20} {m['auc_roc']:>8.4f} {m['pr_auc']:>8.4f} "
                      f"{m['f1_default']:>8.4f} {m['cv_auc_mean']:>8.4f}{tag}")
        print()


def resolve_model_key(tier: str, model_arg: str | None, meta: dict) -> str:
    """Return the exact model key to use, validating it exists in the metadata."""
    if tier == "baseline":
        return "logistic_regression"   # only one model in baseline

    available = list(meta["models"].keys())
    if model_arg is None:
        return meta.get("best_model", available[0])
    if model_arg not in available:
        raise ValueError(
            f"Model '{model_arg}' not found in '{tier}' tier.\n"
            f"  Available: {available}"
        )
    return model_arg


def load_model_and_preprocessor(tier: str, model_key: str, meta: dict):
    """Deserialise preprocessor + chosen model from the tier's saved_models/."""
    tier_dir = TIER_DIRS[tier]

    pre_file = meta["preprocessor_file"]
    pre_path = os.path.join(tier_dir, pre_file)
    if not os.path.exists(pre_path):
        raise FileNotFoundError(f"Preprocessor not found: {pre_path}")

    if tier == "baseline":
        mdl_file = meta["model_file"]
    else:
        mdl_file = meta["models"][model_key]["filename"]

    mdl_path = os.path.join(tier_dir, mdl_file)
    if not os.path.exists(mdl_path):
        raise FileNotFoundError(f"Model file not found: {mdl_path}")

    return joblib.load(mdl_path), joblib.load(pre_path)


def load_input_data(input_csv: str | None, limit: int) -> pd.DataFrame:
    if input_csv:
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"  Source   : {input_csv}  ({len(df):,} rows)")
    else:
        import sqlite3
        conn  = sqlite3.connect(DB_PATH)
        q     = "SELECT * FROM processed_loans" + (f" LIMIT {limit}" if limit > 0 else "")
        df    = pd.read_sql(q, conn)
        conn.close()
        tag   = f" (LIMIT {limit})" if limit else ""
        print(f"  Source   : SQLite → processed_loans{tag}  ({len(df):,} rows)")
    return df


def apply_feature_engineering(df: pd.DataFrame, target_col: str = "loanStatus") -> pd.DataFrame:
    """Replicate the same 5 interaction features added during training."""
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
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


# ─────────────────────────── Main ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CreditPathAI — score loan default risk with a saved model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tier",      default="advanced",
                        choices=["baseline", "advanced"],
                        help="Model tier to use.  [default: advanced]")
    parser.add_argument("--model",     default=None,
                        help="Model key within the tier.  [default: best]")
    parser.add_argument("--input",     default=None,
                        help="Input CSV path.  [default: DB processed_loans]")
    parser.add_argument("--output",    default=None,
                        help="Output CSV path.  [default: predictions/<tier>_<model>.csv]")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Decision threshold P(default).  [default: 0.50]")
    parser.add_argument("--list",      action="store_true",
                        help="List all saved models and exit.")
    parser.add_argument("--limit",     type=int, default=0,
                        help="Max rows from DB (0 = all).  [default: 0]")
    args = parser.parse_args()

    if args.list:
        list_all_models()
        return

    # ── 1. Resolve model ────────────────────────────────────────────────────
    section(f"1 · Loading Saved Model  [{args.tier.upper()}]")
    meta      = load_metadata(args.tier)
    model_key = resolve_model_key(args.tier, args.model, meta)
    print(f"  Tier      : {args.tier}")
    print(f"  Model     : {model_key}")
    print(f"  Trained at: {meta['trained_at']}")

    # Print metrics for chosen model
    if args.tier == "baseline":
        m = meta["metrics"]
    else:
        m = meta["models"][model_key]["metrics"]
    print(f"  AUC-ROC   : {m['auc_roc']}   PR-AUC : {m['pr_auc']}   "
          f"F1(default) : {m['f1_default']}")

    model, preprocessor = load_model_and_preprocessor(args.tier, model_key, meta)
    print("  ✓ Model + preprocessor loaded.")

    # ── 2. Input data ────────────────────────────────────────────────────────
    section("2 · Loading Input Data")
    raw_df = load_input_data(args.input, args.limit)

    # ── 3. Feature engineering ───────────────────────────────────────────────
    section("3 · Feature Engineering")
    X = apply_feature_engineering(raw_df, meta.get("target_col", "loanStatus"))
    print(f"  Columns after engineering : {X.shape[1]}")

    expected = set(meta["feature_names"])
    missing  = expected - set(X.columns)
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required features:\n  {sorted(missing)}")

    X = X[meta["feature_names"]]   # enforce column order

    # ── 4. Preprocess ────────────────────────────────────────────────────────
    section("4 · Preprocessing")
    X_t = preprocessor.transform(X)

    # LightGBM prefers a named DataFrame
    if model_key == "lightgbm":
        X_t = pd.DataFrame(X_t, columns=meta["feature_names"])
    print(f"  Transformed shape : {X_t.shape if hasattr(X_t,'shape') else 'ok'}")

    # ── 5. Predict ───────────────────────────────────────────────────────────
    section("5 · Running Predictions")
    proba = model.predict_proba(X_t)[:, 1]
    pred  = (proba >= args.threshold).astype(int)

    n_default     = int(pred.sum())
    n_non_default = len(pred) - n_default
    print(f"  Threshold              : {args.threshold}")
    print(f"  Total rows scored      : {len(pred):,}")
    print(f"  Predicted Non-Default  : {n_non_default:,}  ({n_non_default/len(pred)*100:.1f}%)")
    print(f"  Predicted Default      : {n_default:,}  ({n_default/len(pred)*100:.1f}%)")

    # ── 6. Build + save output ───────────────────────────────────────────────
    section("6 · Saving Results")
    out_df = raw_df.copy()
    out_df["default_probability"] = np.round(proba, 6)
    out_df["predicted_default"]   = pred
    out_df["risk_band"] = pd.cut(
        proba,
        bins  =[0.0, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["Very Low", "Low", "Medium", "High", "Very High"],
        right =False,
    )

    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    else:
        os.makedirs(PRED_DIR, exist_ok=True)
        out_path = os.path.join(PRED_DIR, f"{args.tier}_{model_key}_predictions.csv")

    out_df.to_csv(out_path, index=False)
    print(f"  ✓ Predictions saved → {out_path}")

    # ── Risk-band summary ────────────────────────────────────────────────────
    print("\n  Risk Band Distribution:")
    bands = (out_df["risk_band"]
             .value_counts()
             .reindex(["Very Low", "Low", "Medium", "High", "Very High"])
             .fillna(0).astype(int))
    for band, count in bands.items():
        pct = count / len(out_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {band:<10} {count:>7,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Done.  →  {out_path}")


if __name__ == "__main__":
    main()
