# CreditPathAI — Recommendation Engine

Delivers **actionable borrower recommendations** by combining XGBoost default-probability predictions with a transparent, policy-driven risk-resolution framework.

---

## Folder Structure

```
recommendation_engine/
├── recommend_engine.py   # Core engine: loads model, scores borrowers, displays results
├── risk_rules.py         # All risk logic: priority tiers, action playbooks, risk flags
├── __init__.py           # Package exports
└── README.md             # This file
```

---

## How It Works

```
Borrower Features
      │
      ▼
 recommend_engine.py
      │
      ├─→ XGBoost model  →  P(default)
      │
      ├─→ Expected Loss = P(default) × Loan Amount
      │
      └─→ risk_rules.py
               │
               ├─→ resolve_priority(expected_loss)  →  Low / Medium / High / Critical
               ├─→ get_action_plan(priority)         →  Team, Channel, Frequency, Action
               ├─→ prob_to_risk_band(prob)           →  Very Low … Very High
               └─→ detect_risk_flags(borrower)       →  Red-flag list
```

---

## Priority Tiers & Resolutions

| Priority | Expected Loss | Action |
|----------|--------------|--------|
| **Low** | < ₹50,000 | Send automated reminder (Email + SMS) |
| **Medium** | ₹50,000 – ₹2,00,000 | Call borrower and discuss repayment plan |
| **High** | ₹2,00,000 – ₹5,00,000 | Assign dedicated recovery agent (field visit) |
| **Critical** | > ₹5,00,000 | Escalate to senior recovery & legal team |

---

## Risk Bands (by Default Probability)

| Band | Probability Range |
|------|-----------------|
| Very Low | < 20% |
| Low | 20% – 40% |
| Medium | 40% – 60% |
| High | 60% – 80% |
| Very High | > 80% |

---

## Usage

### Run from the command line

```powershell
# Score a synthetic demo borrower (default if no flag given)
python recommendation_engine/recommend_engine.py --demo

# Score rows from a CSV file
python recommendation_engine/recommend_engine.py --input path/to/loans.csv

# Score rows from the SQLite database (default limit: 500 rows)
python recommendation_engine/recommend_engine.py --batch

# Save batch results to CSV
python recommendation_engine/recommend_engine.py --batch --output results.csv

# Change the decision threshold (default: 0.50)
python recommendation_engine/recommend_engine.py --demo --threshold 0.35
```

### Use as a Python module

```python
from recommendation_engine import recommend, recommend_batch, load_model

# Load model once
model, preprocessor, feature_names = load_model()

# Score a single borrower
borrower = { "loanAmount": 350000, "interestRate": 19.5, ... }
result = recommend(borrower, model, preprocessor, feature_names)

print(result["priority_level"])       # e.g. "High"
print(result["recommended_action"])   # e.g. "Assign dedicated recovery agent"
print(result["expected_loss"])        # e.g. 241500.0
print(result["risk_flags"])           # e.g. ["High DTI ratio (0.45 ≥ 0.40)", ...]

# Score a batch (DataFrame in, DataFrame out — sorted by expected loss desc)
import pandas as pd
df = pd.read_csv("loans.csv")
scored_df = recommend_batch(df, model, preprocessor, feature_names)
```

---

## Recommendation Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `default_probability` | float | Model P(default), 0–1 |
| `predicted_default` | bool | True if prob ≥ threshold |
| `risk_band` | str | Very Low / Low / Medium / High / Very High |
| `loan_amount` | float | Exposure in ₹ |
| `expected_loss` | float | P(default) × loan amount |
| `priority_level` | str | Low / Medium / High / Critical |
| `assigned_team` | str | Team responsible for recovery |
| `recovery_channel` | str | Contact channel (Email, Call, Field Visit, Legal) |
| `follow_up_frequency` | str | How often to follow up |
| `legal_action` | bool | Whether legal action is required |
| `recommended_action` | str | Plain-English action for the recovery officer |
| `escalation_notes` | str | Additional escalation guidance |
| `risk_flags` | list[str] | Qualitative red flags detected from borrower features |

---

## Customising Risk Policy

All policy decisions are in **`risk_rules.py`** — no need to touch the engine:

```python
# risk_rules.py

# Adjust priority tiers
LOSS_BANDS = [
    (50_000,       "Low"),
    (200_000,      "Medium"),
    (500_000,      "High"),
    (float("inf"), "Critical"),
]

# Adjust action playbooks
ACTION_PLAYBOOKS["Medium"]["follow_up_frequency"] = "Twice a week"

# Adjust risk-flag sensitivity
HIGH_DTI_THRESHOLD = 0.35   # flag borrowers with DTI ≥ 35% instead of 40%
```

---

## Model Artefacts

Located at `training/advanced/saved_models/`:

| File | Description |
|------|-------------|
| `xgboost.joblib` | Trained XGBoost classifier |
| `preprocessor.joblib` | Fitted ColumnTransformer (scaling + encoding) |
| `metadata.json` | Feature names, training date, metrics |
