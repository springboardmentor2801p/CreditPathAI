import pandas as pd
import sqlite3
import json
from database import get_conn, init_db, hash_password
from credit_recommendation_engine import expected_loss_engine
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any

# Ensure DB and tables exist on startup
init_db()


app = FastAPI(title="CreditPath AI Risk Scoring & Platform API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_borrower_input(raw_input):
    # Handle numeric age by bucketing it into strings expected by the ML model
    if 'age' in raw_input and isinstance(raw_input['age'], (int, float)):
        age = raw_input['age']
        if age < 25: raw_input['age'] = '<25'
        elif age <= 34: raw_input['age'] = '25-34'
        elif age <= 44: raw_input['age'] = '35-44'
        elif age <= 54: raw_input['age'] = '45-54'
        elif age <= 64: raw_input['age'] = '55-64'
        elif age <= 74: raw_input['age'] = '65-74'
        else: raw_input['age'] = '>74'

    df = pd.DataFrame([raw_input])
    if 'credit_score' in df.columns:
        df.rename(columns={'credit_score': 'Credit_Score'}, inplace=True)
    if 'property_value' in df:
        pv_med = df['property_value'].median()
        df['property_value'] = df['property_value'].fillna(pv_med if not pd.isna(pv_med) else 0)
    if 'loan_amount' in df and 'property_value' in df:
        df['LTV'] = (df['loan_amount'] / df['property_value']) * 100
    num_cols = ['Upfront_charges','Interest_rate_spread','rate_of_interest','dtir1','income','term']
    for col in num_cols:
        if col in df:
            col_med = df[col].median()
            df[col] = df[col].fillna(col_med if not pd.isna(col_med) else 0)
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        mode_val = df[col].mode()
        df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
    binary_map = {
        'loan_limit': {'cf':0,'ncf':1},
        'approv_in_adv': {'nopre':0,'pre':1},
        'open_credit': {'nopc':0,'opc':1},
        'business_or_commercial': {'nob/c':0,'b/c':1},
        'Neg_ammortization': {'not_neg':0,'neg_amm':1},
        'interest_only': {'not_int':0,'int_only':1},
        'lump_sum_payment': {'not_lpsm':0,'lpsm':1},
        'Credit_Worthiness': {'l1':1,'l2':0}
    }
    for col, mapping in binary_map.items():
        if col in df:
            df[col] = df[col].map(mapping)
    age_map = {'<25':1,'25-34':2,'35-44':3,'45-54':4,'55-64':5,'65-74':6,'>74':7}
    if 'age' in df:
        df['age'] = df['age'].map(age_map)
    df = pd.get_dummies(df, drop_first=False)
    if 'income' in df and 'loan_amount' in df:
        df['loan_to_income'] = df['loan_amount'] / df['income']
    if 'LTV' in df:
        df['high_LTV_flag'] = (df['LTV'] > 90).astype(int)
    if 'Credit_Score' in df:
        df['low_credit_flag'] = (df['Credit_Score'] < 650).astype(int)
    if 'dtir1' in df:
        df['high_dti_flag'] = (df['dtir1'] > 45).astype(int)
    if 'low_credit_flag' in df and 'high_LTV_flag' in df:
        df['risk_interaction'] = df['low_credit_flag'] * df['high_LTV_flag']
    if 'low_credit_flag' in df and 'high_dti_flag' in df:
        df['credit_dti_risk'] = df['low_credit_flag'] * df['high_dti_flag']
    if 'high_LTV_flag' in df and 'high_dti_flag' in df:
        df['ltv_dti_risk'] = df['high_LTV_flag'] * df['high_dti_flag']
    if 'low_credit_flag' in df and 'high_LTV_flag' in df and 'high_dti_flag' in df:
        df['triple_risk_flag'] = ((df['low_credit_flag'] + df['high_LTV_flag'] + df['high_dti_flag']) == 3).astype(int)
    if 'Credit_Score' in df:
        df['credit_bucket'] = pd.cut(df['Credit_Score'], bins=[0, 580, 669, 739, 799, 900], labels=[0, 1, 2, 3, 4]).astype(float)
    if 'LTV' in df:
        df['ltv_bucket'] = pd.cut(df['LTV'], bins=[0, 60, 80, 90, 100, 200], labels=[0, 1, 2, 3, 4]).astype(float)
    feature_names = pd.read_csv('model_features.csv', header=None).iloc[0].tolist()
    missing_cols = [col for col in feature_names if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    df = df[feature_names]
    return df.values.reshape(1, -1)

class BorrowerInput(BaseModel):
    fullName: Optional[str] = None
    income: float
    credit_score: float
    loan_amount: float
    property_value: float
    age: Any
    debt_ratio: float = None
    missed_payments: int = None
    term: int = None
    region: str = None
    loan_type: str = None
    employment_status: str = None
    Credit_Worthiness: str = None
    days_overdue: int = 0

@app.post("/risk-score")
def get_risk_score(data: BorrowerInput, x_user_id: Optional[str] = Header(None)):
    input_data = data.dict()
    feature_vector = preprocess_borrower_input(input_data)
    result = expected_loss_engine({
        **input_data,
        'feature_vector': feature_vector,
        'loan_type': input_data.get('loan_type', 'secured'),
        'age': input_data.get('age', 35)
    })
    
    # Note: we no longer auto-save to institution_cases here.
    # The lender explicitly saves via POST /api/cases after reviewing
    # the decision (APPROVE → active, CONDITIONAL → conditional, REJECT → rejected).
    # This keeps the portfolio clean and decision-intentional.

    # Only write a lightweight audit entry so the action is tracked.
    try:
        user_id = get_user_from_token(x_user_id)
        borrower_name = (data.fullName or f"Borrower {int(data.credit_score)}").strip()
        conn = get_conn()
        c = conn.cursor()
        c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
            (user_id, "Risk Evaluated",
             f"{borrower_name} — PD {result['default_probability']*100:.1f}% → {result.get('loan_decision','N/A')}",
             "evaluation"))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Audit log error:", str(e))

    return result



# ============================================================
# PORTFOLIO CASE MANAGEMENT
# ============================================================

class CaseSaveBody(BaseModel):
    borrower_name: str
    loan_amount: float
    outstanding: float
    credit_score: float
    days_overdue: int = 0
    default_probability: float
    priority: str
    recommended_action: str
    assigned_team: str
    loan_decision: Optional[str] = None
    notes: Optional[str] = None
    # Status reflects the lender's decision:
    #   'active'      — approved, fully tracked in recovery/portfolio
    #   'conditional' — conditional approval, pending conditions being met
    #   'rejected'    — declined, archived for regulatory record
    status: str = "active"


@app.post("/api/cases")
def save_case(body: CaseSaveBody, x_user_id: Optional[str] = Header(None)):
    """Lender explicitly saves a loan/NPA into the tracked portfolio. Assigned team is required."""
    import uuid
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    acc_id = "ACC" + str(uuid.uuid4())[:8].upper()
    # Ensure borrower_name is a non-empty, non-numeric string
    borrower_name = body.borrower_name
    if not borrower_name or borrower_name.strip() == "" or borrower_name.strip().isdigit():
        borrower_name = "Unknown Borrower"
    # Force assigned_team to be required and non-empty
    assigned_team = (body.assigned_team or "").strip()
    if not assigned_team:
        conn.close()
        raise HTTPException(status_code=400, detail="assigned_team is required and cannot be empty")
    c.execute("""INSERT INTO institution_cases
        (account_id, lender_id, borrower_name, loan_amount, outstanding, credit_score,
         days_overdue, default_probability, priority, recommended_action, assigned_team, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (acc_id, user_id, borrower_name, body.loan_amount, body.outstanding,
         body.credit_score, body.days_overdue, body.default_probability,
         body.priority, body.recommended_action, assigned_team,
         body.status))
    case_id = c.lastrowid
    decision_note = f" — Decision: {body.loan_decision}" if body.loan_decision else ""
    action_label = {
        "active":      "Approved & Tracked",
        "conditional": "Saved as Conditional",
        "rejected":    "Declined & Archived",
    }.get(body.status, "Saved")
    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
        (user_id, "Case Saved", f"Case #{case_id} ({acc_id}) — {body.borrower_name} → {action_label}{decision_note}", "institution_cases"))
    conn.commit()
    conn.close()
    return {"ok": True, "account_id": acc_id, "case_id": case_id, "status": body.status}

@app.get("/api/cases")
def list_cases(team: Optional[str] = None, priority: Optional[str] = None, status: Optional[str] = None, x_user_id: Optional[str] = Header(None)):
    """List cases, optionally filtered by team/priority/status, always filtered by lender_id."""
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    q = "SELECT * FROM institution_cases WHERE lender_id = ?"
    params = [user_id]
    
    if team:
        q += " AND assigned_team = ?"
        params.append(team)
    if priority:
        q += " AND LOWER(priority) = LOWER(?)"
        params.append(priority)
    if status:
        q += " AND status = ?"
        params.append(status)
        
    q += " ORDER BY default_probability DESC, created_at DESC LIMIT 50"
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return {"cases": [dict(r) for r in rows]}


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM institution_cases WHERE id = ? AND lender_id = ?", (case_id, user_id))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Case not found or access denied")
    case_data = dict(row)
    # If resolved, clear or update recommended_action
    if case_data.get("status") == "resolved":
        case_data["recommended_action"] = None  # or set to "No further action required"
    return case_data

@app.patch("/api/cases/{case_id}/reassign")
def reassign_case(case_id: int, body: dict, x_user_id: Optional[str] = Header(None)):
    new_team = body.get("team")
    new_priority = body.get("priority")
    if not new_team:
        raise HTTPException(status_code=400, detail="team required")
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    # verify ownership before update
    c.execute("UPDATE institution_cases SET assigned_team=?, priority=COALESCE(?,priority), updated_at=CURRENT_TIMESTAMP WHERE id=? AND lender_id=?",
              (new_team, new_priority, case_id, user_id))
    if c.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found or access denied")
    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
              (user_id, "Case Reassigned", f"Case #{case_id} reassigned to {new_team}", "institution_cases"))
    conn.commit()
    conn.close()
    return {"ok": True, "case_id": case_id, "team": new_team}

@app.patch("/api/cases/{case_id}/approve")
def approve_case(case_id: int, body: dict, x_user_id: Optional[str] = Header(None)):
    """Promote a conditional loan application to active (lender confirms conditions met)."""
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT status, borrower_name FROM institution_cases WHERE id = ? AND lender_id = ?", (case_id, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found or access denied")
    if row["status"] not in ("conditional",):
        conn.close()
        raise HTTPException(status_code=400, detail=f"Cannot approve case with status '{row['status']}'")
    c.execute(
        "UPDATE institution_cases SET status='active', updated_at=CURRENT_TIMESTAMP WHERE id=? AND lender_id=?",
        (case_id, user_id)
    )
    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
        (user_id, "Loan Approved",
         f"Case #{case_id} ({row['borrower_name']}) — conditional → active after conditions met",
         "institution_cases"))
    conn.commit()
    conn.close()
    return {"ok": True, "case_id": case_id, "status": "active"}



class PreApprovalInput(BaseModel):
    loanPurpose: str
    loanAmount: float
    annualIncome: float
    monthlyDebt: float
    creditScore: float
    employmentStatus: str
    propertyValue: float = 0.0
    term: int = 180

@app.post("/api/evaluate-loan")
def evaluate_loan(data: PreApprovalInput):
    # Convert inputs to expected dictionary layout
    input_data = {
        "income": data.annualIncome,
        "credit_score": data.creditScore,
        "loan_amount": data.loanAmount,
        "property_value": data.propertyValue,
        "term": data.term,
        "age": "35-44",
        "days_overdue": 0
    }
    feature_vector = preprocess_borrower_input(input_data)
    result = expected_loss_engine({
        **input_data,
        'feature_vector': feature_vector
    })
    
    # Process Borrower-facing metrics from ML response
    default_prob = result["default_probability"]
    approval_prob = (1.0 - default_prob) * 100
    
    dti = (data.monthlyDebt * 12) / max(data.annualIncome, 1)
    
    if approval_prob >= 95:
        tier = "EXCELLENT"
        tierColor = "emerald"
        estRate = "5.5% - 7.0%"
    elif approval_prob >= 80:
        tier = "GOOD"
        tierColor = "cyan"
        estRate = "8.5% - 12.0%"
    elif approval_prob >= 50:
        tier = "MODERATE"
        tierColor = "amber"
        estRate = "13.0% - 18.0%"
    else:
        tier = "HIGH_RISK"
        tierColor = "rose"
        estRate = "19.0% - 24.0%+"

    recs = []
    
    if approval_prob >= 95:
        recs.append("Profile optimization complete. You qualify for the best available rates.")
    else:
        # Provide targeted paths forward based on proximity to next tier
        if approval_prob < 50:
            recs.append(f"High risk classification. Needs significant optimization before formal application.")
        elif approval_prob < 80:
            recs.append(f"Moderate tier. You are {(80 - approval_prob):.1f}% away from the GOOD tier brackets.")

    # DTI specific math
    if dti > 0.4:
        target_monthly_debt = (0.35 * max(data.annualIncome, 1)) / 12
        reduction_needed = data.monthlyDebt - target_monthly_debt
        if reduction_needed > 0:
            recs.append(f"DTI is {dti*100:.1f}%. Reduce monthly debt by ₹{reduction_needed:,.0f} to reach the secure 35% threshold.")
        else:
            recs.append("DTI Ratio is critical. Pay down debt to improve odds.")
    else:
        recs.append(f"DTI ({dti*100:.1f}%) is healthy. Maintain current balances.")

    # Credit Score specific math
    if data.creditScore < 650:
        points_needed = 650 - int(data.creditScore)
        recs.append(f"Credit Base is weak. Adding {points_needed} points to your score will qualify you for standard market rates.")
    elif data.creditScore < 750:
        points_needed = 750 - int(data.creditScore)
        recs.append(f"Credit Base is strong, but an additional {points_needed} points unlocks premium luxury tiers.")
    else:
        recs.append("Credit score is excellent. You qualify for minimum-rate interest configurations.")
        
    # LTV specific math
    ltv = (data.loanAmount / max(data.propertyValue, 1)) * 100
    if ltv > 85:
        target_loan = data.propertyValue * 0.80
        down_payment_gap = data.loanAmount - target_loan
        if down_payment_gap > 0:
            recs.append(f"High LTV ({ltv:.1f}%). Increase down payment by ₹{down_payment_gap:,.0f} to push LTV to safe 80%.")
        else:
            recs.append("Property value insufficient for requested loan volume.")

    return {
        "approvalProb": round(approval_prob, 1),
        "estRate": estRate,
        "tier": tier,
        "tierColor": tierColor,
        "dti": round(dti * 100, 1),
        "recommendations": recs
    }



# ============================================================
# AUTH MODELS & ENDPOINTS
# ============================================================

class LoginInput(BaseModel):
    email: str
    password: str

class RegisterInput(BaseModel):
    email: str
    password: str
    full_name: str
    role: str  # 'borrower' or 'institution'
    # Additional fields for borrower registration
    credit_score: Optional[int] = None
    annual_income: Optional[float] = None
    employment_status: Optional[str] = None

def get_user_from_token(x_user_id: Optional[str] = Header(None)):
    """Simple user ID from header — production would use JWT."""
    if not x_user_id:
        return None
    try:
        return int(x_user_id)
    except:
        return None

@app.post("/api/auth/login")
def login(data: LoginInput):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (data.email,))
    user = c.fetchone()
    conn.close()
    if not user or user["password_hash"] != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {
        "user_id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"]
    }

@app.post("/api/auth/register")
def register(data: RegisterInput):
    if data.role not in ("borrower", "institution"):
        raise HTTPException(status_code=400, detail="Invalid role")
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password_hash, role, full_name) VALUES (?,?,?,?)",
            (data.email, hash_password(data.password), data.role, data.full_name))
        user_id = c.lastrowid
        if data.role == "borrower":
            # Require important fields for borrower
            if data.credit_score is None or data.annual_income is None or not data.employment_status:
                conn.close()
                raise HTTPException(status_code=400, detail="Missing required borrower registration fields.")
            c.execute("""
                INSERT INTO borrower_profiles (user_id, credit_score, annual_income, employment_status, next_payment_due, next_payment_days)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (user_id, data.credit_score, data.annual_income, data.employment_status, 0, 30)
            )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Email already registered")
    conn.close()
    return {"user_id": user_id, "role": data.role, "full_name": data.full_name, "email": data.email}

@app.get("/api/auth/profile")
def get_auth_profile(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, email, full_name, role FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(user)

# ============================================================
# BORROWER PROFILE — full financial profile GET + PUT
# ============================================================

@app.get("/api/borrower-profile")
def get_borrower_profile(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT u.full_name, u.email, u.role, bp.* FROM users u LEFT JOIN borrower_profiles bp ON u.id = bp.user_id WHERE u.id = ?", (user_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")
    c.execute("SELECT * FROM borrower_loans WHERE user_id = ?", (user_id,))
    loans = [dict(l) for l in c.fetchall()]
    conn.close()
    
    # Calculate next payment due dynamically based on active loans
    next_due_calc = 0
    next_days = 30 # Default if no loans
    for l in loans:
        if l["status"] != "paid_off" and l["term_months"] > 0:
            monthly_rate = (l["rate"] / 100) / 12
            if monthly_rate > 0:
                emi = l["loan_amount"] * monthly_rate * ((1 + monthly_rate) ** l["term_months"]) / (((1 + monthly_rate) ** l["term_months"]) - 1)
            else:
                emi = l["loan_amount"] / l["term_months"]
            next_due_calc += emi
            next_days = min(next_days, 15) # Example arbitrary simple logic
            
    row_dict = dict(row)
    row_dict["next_payment_due"] = int(next_due_calc)
    row_dict["next_payment_days"] = next_days
    
    monthly_income = max(row_dict.get("annual_income", 500000) / 12, 1)
    row_dict["debt_to_income"] = (next_due_calc / monthly_income) * 100
    
    return {**row_dict, "loans": loans}

class ProfileUpdateInput(BaseModel):
    full_name: Optional[str] = None
    credit_score: Optional[int] = None
    annual_income: Optional[float] = None
    total_debt: Optional[float] = None
    employment_status: Optional[str] = None

@app.put("/api/borrower-profile")
def update_borrower_profile(data: ProfileUpdateInput, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_conn()
    c = conn.cursor()

    if data.full_name:
        c.execute("UPDATE users SET full_name = ? WHERE id = ?", (data.full_name, user_id))

    profile_updates = {k: v for k, v in data.dict().items()
                       if v is not None and k != 'full_name'}

    if profile_updates:
        # Recalculate DTI if income/debt changed
        if 'annual_income' in profile_updates or 'total_debt' in profile_updates:
            c.execute("SELECT annual_income, total_debt FROM borrower_profiles WHERE user_id = ?", (user_id,))
            existing = dict(c.fetchone() or {})
            income = profile_updates.get('annual_income', existing.get('annual_income', 1))
            debt = profile_updates.get('total_debt', existing.get('total_debt', 0))
            profile_updates['debt_to_income'] = round((debt / max(income, 1)) * 100, 1)

        # Upsert profile
        c.execute("SELECT id FROM borrower_profiles WHERE user_id = ?", (user_id,))
        exists = c.fetchone()
        if exists:
            set_clause = ", ".join(f"{k}=?" for k in profile_updates)
            c.execute(f"UPDATE borrower_profiles SET {set_clause}, updated_at=datetime('now') WHERE user_id=?",
                      list(profile_updates.values()) + [user_id])
        else:
            cols = ", ".join(profile_updates.keys())
            vals = ", ".join("?" * len(profile_updates))
            c.execute(f"INSERT INTO borrower_profiles (user_id, {cols}) VALUES (?, {vals})",
                      [user_id] + list(profile_updates.values()))

    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
              (user_id, "Profile Updated", f"Updated: {list(data.dict(exclude_none=True).keys())}", "profile"))
    conn.commit()
    conn.close()
    return {"status": "updated"}

class LoanInput(BaseModel):
    loan_name: str
    loan_amount: float
    outstanding: float
    rate: float
    term_months: int
    status: str = "current"
    status_type: str = "good"

@app.post("/api/borrower-loans")
def add_loan(data: LoanInput, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO borrower_loans
        (user_id, loan_name, loan_amount, outstanding, rate, term_months, status, status_type)
        VALUES (?,?,?,?,?,?,?,?)""",
        (user_id, data.loan_name, data.loan_amount, data.outstanding,
         data.rate, data.term_months, data.status, data.status_type))
    # Recalculate total debt
    c.execute("SELECT SUM(outstanding) as total FROM borrower_loans WHERE user_id=?", (user_id,))
    total = c.fetchone()["total"] or 0
    c.execute("UPDATE borrower_profiles SET total_debt=?, updated_at=datetime('now') WHERE user_id=?",
              (total, user_id))
    conn.commit()
    conn.close()
    return {"status": "added"}

@app.delete("/api/borrower-loans/{loan_id}")
def delete_loan(loan_id: int, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM borrower_loans WHERE id=? AND user_id=?", (loan_id, user_id))
    c.execute("SELECT SUM(outstanding) as total FROM borrower_loans WHERE user_id=?", (user_id,))
    total = c.fetchone()["total"] or 0
    c.execute("UPDATE borrower_profiles SET total_debt=?, updated_at=datetime('now') WHERE user_id=?",
              (total, user_id))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# ============================================================
# BORROWER DASHBOARD (real DB data)
# ============================================================

@app.get("/api/borrower-dashboard")
def get_borrower_dashboard_data(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT u.full_name, bp.* 
        FROM users u 
        LEFT JOIN borrower_profiles bp ON u.id = bp.user_id 
        WHERE u.id = ?
    """, (user_id,))
    profile = c.fetchone()
    if not profile:
        conn.close()
        raise HTTPException(status_code=404, detail="Profile not found")

    c.execute("SELECT * FROM borrower_loans WHERE user_id = ?", (user_id,))
    loans = c.fetchall()

    c.execute("""SELECT action, details, created_at FROM audit_log WHERE user_id = ?
                 ORDER BY id DESC LIMIT 10""", (user_id,))
    audit = c.fetchall()
    conn.close()

    # Credit score history
    try:
        cs_history = json.loads(profile["credit_score_history"])
        if isinstance(cs_history, list):
            cs_history = {"labels":[],"values":[]}
    except Exception:
        cs_history = {"labels":[],"values":[]}
        
    credit_score = profile["credit_score"]

    # Debt structure from actual loans
    debt_labels = [l["loan_name"] for l in loans]
    debt_values = [l["outstanding"] for l in loans]
    total_debt = sum(debt_values)

    # Calculate next payment due dynamically based on active loans
    next_due_calc = 0
    next_days = 30 # Default if no loans
    for l in loans:
        if l["status"] != "paid_off" and l["term_months"] > 0:
            monthly_rate = (l["rate"] / 100) / 12
            if monthly_rate > 0:
                emi = l["loan_amount"] * monthly_rate * ((1 + monthly_rate) ** l["term_months"]) / (((1 + monthly_rate) ** l["term_months"]) - 1)
            else:
                emi = l["loan_amount"] / l["term_months"]
            next_due_calc += emi
            next_days = min(next_days, 15) # Example arbitrary simple logic
    next_due = int(next_due_calc)

    # Stats - Dynamic DTI
    monthly_income = max(profile["annual_income"] / 12, 1)
    dti = (next_due_calc / monthly_income) * 100

    # Previous score for change indicator
    prev_score = cs_history["values"][-2] if len(cs_history.get("values", [])) > 1 else credit_score
    score_change = credit_score - prev_score

    stats = [
        {"name": "CREDIT SCORE (CIBIL)", "value": str(credit_score),
         "change": f"{'+'if score_change>=0 else ''}{score_change} Pts", "changeType": "positive" if score_change >= 0 else "negative",
         "iconType": "Activity", "color": "text-emerald-400", "bg": "bg-emerald-500/10 border-emerald-500/20"},
        {"name": "TOTAL OUTSTANDING DEBT", "value": f"₹{total_debt:,.0f}",
         "change": "Active Loans", "changeType": "neutral",
         "iconType": "DollarSign", "color": "text-cyan-400", "bg": "bg-cyan-500/10 border-cyan-500/20"},
        {"name": "DEBT TO INCOME (DTI)", "value": f"{dti:.1f}%",
         "change": "Good" if dti < 36 else "High", "changeType": "positive" if dti < 36 else "negative",
         "iconType": "TrendingUp", "color": "text-amber-400" if dti >= 36 else "text-emerald-400",
         "bg": "bg-amber-500/10 border-amber-500/20" if dti >= 36 else "bg-emerald-500/10 border-emerald-500/20"},
        {"name": "NEXT PAYMENT DUE", "value": f"₹{next_due:,.0f}",
         "change": f"{next_days} Days", "changeType": "neutral" if next_days > 7 else "negative",
         "iconType": "AlertTriangle", "color": "text-cyan-400", "bg": "bg-cyan-500/10 border-cyan-500/20"},
    ]

    # Smart recommendations based on actual data
    recs = []
    if dti > 40:
        recs.append({"id": 1, "title": "Reduce Monthly Debt Burden",
                     "desc": f"Your DTI is {dti:.1f}%. Aim below 36% by paying down high-interest debt first.",
                     "impact": "High Priority", "type": "Action Needed"})
    if credit_score < 700:
        recs.append({"id": 2, "title": "Improve Your CIBIL Score",
                     "desc": f"Your score is {credit_score}. Timely payments for 6 months can add 30-50 points.",
                     "impact": "+40 Pts", "type": "High Impact"})
    for loan in loans:
        if loan["status_type"] == "warning":
            recs.append({"id": 3, "title": f"High Utilization: {loan['loan_name']}",
                         "desc": f"Outstanding ₹{loan['outstanding']:,.0f}. Keep below 30% utilization.",
                         "impact": "Prevent Drop", "type": "Action Needed"})
            break
    if not recs:
        recs.append({"id": 1, "title": "Excellent Financial Standing",
                     "desc": "Your profile is strong. Consider a Home Loan Pre-Approval now.",
                     "impact": "Ready", "type": "High Impact"})

    # Liabilities from DB
    liabilities = []
    for loan in loans:
        liabilities.append({
            "id": loan["id"],
            "name": loan["loan_name"],
            "details": f"Rate: {loan['rate']}% // Term: {loan['term_months']}M",
            "amount": f"₹{loan['outstanding']:,.0f}",
            "status": loan["status"].replace("_", " ").title(),
            "statusType": loan["status_type"]
        })

    # Flatten all required fields to top level for frontend compatibility
    return {
        "full_name": profile["full_name"],
        "credit_score": profile["credit_score"],
        "annual_income": profile["annual_income"],
        "next_payment_due": next_due,
        "next_payment_days": next_days,
        "employment_status": profile["employment_status"],
        "profile": {
            "full_name": profile["full_name"],
            "credit_score": profile["credit_score"],
            "annual_income": profile["annual_income"],
            "next_payment_due": next_due,
            "next_payment_days": next_days,
            "employment_status": profile["employment_status"]
        },
        "stats": stats,
        "creditTrajectory": cs_history,
        "debtStructure": {"values": debt_values, "labels": debt_labels},
        "recommendations": recs,
        "liabilities": liabilities
    }

# ============================================================
# INSTITUTION DASHBOARD (real DB data)
# ============================================================

@app.get("/api/dashboard")
def get_dashboard_data(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) as total, SUM(outstanding) as total_outstanding FROM institution_cases WHERE status='active' AND lender_id = ?", (user_id,))
    summary_row = c.fetchone()
    summary = {
        'total': summary_row['total'] if summary_row and summary_row['total'] is not None else 0,
        'total_outstanding': summary_row['total_outstanding'] if summary_row and summary_row['total_outstanding'] is not None else 0,
    }

    c.execute("SELECT COUNT(*) as cnt FROM institution_cases WHERE priority IN ('High','Critical') AND status='active' AND lender_id = ?", (user_id,))
    high_risk_row = c.fetchone()
    high_risk = high_risk_row['cnt'] if high_risk_row and high_risk_row['cnt'] is not None else 0

    c.execute("SELECT AVG(default_probability) as avg_prob FROM institution_cases WHERE status='active' AND lender_id = ?", (user_id,))
    avg_prob_row = c.fetchone()
    avg_prob = avg_prob_row['avg_prob'] if avg_prob_row and avg_prob_row['avg_prob'] is not None else 0

    c.execute("SELECT * FROM institution_cases WHERE lender_id = ? ORDER BY default_probability DESC LIMIT 5", (user_id,))
    top_cases = c.fetchall()

    c.execute("""SELECT action, details, created_at FROM audit_log WHERE user_id = ? ORDER BY id DESC LIMIT 5""", (user_id,))
    recent_logs = c.fetchall()
    conn.close()

    # Calculate recovery rate: total resolved / (resolved + active)
    c = get_conn().cursor()
    c.execute("SELECT SUM(outstanding) as resolved_sum FROM institution_cases WHERE status='resolved' AND lender_id = ?", (user_id,))
    resolved_row = c.fetchone()
    resolved_sum = resolved_row['resolved_sum'] if resolved_row and resolved_row['resolved_sum'] is not None else 0

    c.execute("SELECT SUM(outstanding) as active_sum FROM institution_cases WHERE status='active' AND lender_id = ?", (user_id,))
    active_row = c.fetchone()
    active_sum = active_row['active_sum'] if active_row and active_row['active_sum'] is not None else 0

    c.execute("SELECT SUM(loan_amount) as resolved_loan_sum FROM institution_cases WHERE status='resolved' AND lender_id = ?", (user_id,))
    resolved_loan_row = c.fetchone()
    resolved_loan_sum = resolved_loan_row['resolved_loan_sum'] if resolved_loan_row and resolved_loan_row['resolved_loan_sum'] is not None else 0

    # Recovery Rate = (Total Recovered) / (Total Recovered + Total Outstanding) * 100
    # Total Recovered = resolved_loan_sum - resolved_sum
    total_recovered = resolved_loan_sum - resolved_sum
    total_due = total_recovered + active_sum
    recovery_rate = (total_recovered / total_due * 100) if total_due > 0 else 0

    stats = [
        {"name": "TOTAL OUTSTANDING DEBT", "value": f"₹{summary['total_outstanding']:,.0f}",
         "change": f"{summary['total']} Active Cases", "changeType": "neutral",
         "iconType": "DollarSign", "color": "text-cyan-400", "bg": "bg-cyan-500/10 border-cyan-500/20"},
        {"name": "AVG RISK SCORE", "value": f"{avg_prob*100:.1f}%",
         "change": "Default Prob", "changeType": "negative" if avg_prob > 0.4 else "positive",
         "iconType": "Activity", "color": "text-amber-400", "bg": "bg-amber-500/10 border-amber-500/20"},
        {"name": "HIGH RISK CASES", "value": str(high_risk),
         "change": "Needs Action", "changeType": "negative" if high_risk > 3 else "positive",
         "iconType": "AlertTriangle", "color": "text-rose-400", "bg": "bg-rose-500/10 border-rose-500/20"},
        {"name": "RECOVERY RATE", "value": f"{recovery_rate:.1f}%",
         "change": "+2.1% MoM", "changeType": "positive",
         "iconType": "TrendingUp", "color": "text-emerald-400", "bg": "bg-emerald-500/10 border-emerald-500/20"},
    ]

    alerts = [
        {"id": i+1, "borrower": case["borrower_name"], "risk": case["priority"],
         "amount": f"₹{case['outstanding']:,.0f}", "action": case["recommended_action"],
         "time": "Live", "riskType": case["priority"]}
        for i, case in enumerate(top_cases)
    ]

    recent_alerts = [
        {"id": i+1, "borrower": log["action"], "risk": "Audit",
         "amount": "", "action": log["details"], "time": log["created_at"][:10], "riskType": "Low"}
        for i, log in enumerate(recent_logs)
    ]

    # Only provide recoveryVectorData if real data exists (example: empty for new accounts)
    recovery_vector = {"x": [], "y": []}
    # TODO: Populate recovery_vector from real DB data if available
    return {
        "stats": stats,
        "recentAlerts": alerts[:3] + recent_alerts[:2],
        "riskMatrixData": {
            "x": ["Critical", "High", "Medium", "Low"],
            "y": [
                len([c for c in top_cases if c["priority"] == "Critical"]),
                len([c for c in top_cases if c["priority"] == "High"]),
                len([c for c in top_cases if c["priority"] == "Medium"]),
                len([c for c in top_cases if c["priority"] == "Low"]),
            ]
        },
        "recoveryVectorData": recovery_vector
    }

# ============================================================
# INSTITUTION ANALYTICS (real DB data)
# ============================================================

@app.get("/api/analytics")
def get_analytics_data(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()

    # Priority distribution
    c.execute("SELECT priority, COUNT(*) as cnt FROM institution_cases WHERE lender_id = ? GROUP BY priority", (user_id,))
    priority_rows = {r["priority"]: r["cnt"] for r in c.fetchall()}

    # Average default probability
    c.execute("SELECT AVG(default_probability) as avg FROM institution_cases WHERE status='active' AND lender_id = ?", (user_id,))
    avg_prob = c.fetchone()["avg"] or 0

    # Total outstanding and expected loss by team
    c.execute("""
        SELECT assigned_team,
               COUNT(*) as cnt,
               SUM(outstanding) as total_outstanding,
               AVG(default_probability) as avg_pd
        FROM institution_cases
        WHERE lender_id = ?
        GROUP BY assigned_team
    """, (user_id,))
    team_rows = c.fetchall()

    # Cases resolved vs active
    c.execute("SELECT status, COUNT(*) as cnt FROM institution_cases WHERE lender_id = ? GROUP BY status", (user_id,))
    status_rows = {r["status"]: r["cnt"] for r in c.fetchall()}
    total_cases = sum(status_rows.values())
    resolved = status_rows.get("resolved", 0)
    resolution_rate = round(resolved / max(total_cases, 1) * 100, 1)

    # Daily evaluation trend (last 7 days activity in audit_log)
    c.execute("""
        SELECT substr(created_at,1,10) as day, COUNT(*) as cnt
        FROM audit_log
        WHERE user_id = ? AND action IN ('Risk Evaluated', 'Risk Scoring')
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """, (user_id,))
    trend_raw = c.fetchall()
    trend_days  = [r["day"] for r in reversed(trend_raw)]
    trend_count = [r["cnt"] for r in reversed(trend_raw)]

    # Outstanding amount and case count per team (for bar chart)
    team_names = [r["assigned_team"] for r in team_rows]
    team_counts = [r["cnt"] for r in team_rows]
    team_outstanding = [round(r["total_outstanding"] / 1e5, 1) if r["total_outstanding"] else 0 for r in team_rows]  # in lakhs

    conn.close()

    return {
        "defaultDistribution": {
            "values": [
                priority_rows.get("Low", 0),
                priority_rows.get("Medium", 0),
                priority_rows.get("High", 0),
                priority_rows.get("Critical", 0)
            ],
            "labels": ["Low Risk", "Medium Risk", "High Risk", "Critical"]
        },
        "teamExposure": {
            "teams": team_names,
            "counts": team_counts,
            "outstanding_lakhs": team_outstanding
        },
        "evaluationTrend": {
            "days": trend_days,
            "counts": trend_count
        },
        "kpis": {
            "totalCases": total_cases,
            "activeCases": status_rows.get("active", 0),
            "resolvedCases": resolved,
            "resolutionRate": f"{resolution_rate}%",
            "avgDefaultProb": f"{avg_prob*100:.1f}%",
            "totalOutstanding": sum(r["total_outstanding"] or 0 for r in team_rows),
        }
    }

# ============================================================
# INSTITUTION RECOVERY ACTIONS (real DB data)
# ============================================================


@app.get("/api/recovery")
def get_recovery_data(
    priority: Optional[str] = None,
    team: Optional[str] = None,
    status: Optional[str] = None,
    x_user_id: Optional[str] = Header(None)
):
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()


    query = "SELECT * FROM institution_cases WHERE lender_id = ? AND status NOT IN ('rejected', 'conditional')"
    params = [user_id]
    if priority and priority != "all":
        query += " AND LOWER(priority) = LOWER(?)"
        params.append(priority)
    if team and team != "all":
        query += " AND assigned_team = ?"
        params.append(team)
    if status == "pending":
        query += " AND days_overdue > 0"
    elif status == "completed":
        query += " AND status = 'resolved'"

    query += " ORDER BY default_probability DESC LIMIT 30"

    c.execute(query, params)
    cases = c.fetchall()

    # Summary stats (all cases, for this lender only)
    c.execute("SELECT COUNT(*) as cnt FROM institution_cases WHERE status = 'resolved' AND lender_id = ?", (user_id,))
    completed_count = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM institution_cases WHERE lender_id = ?", (user_id,))
    total_count = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM institution_cases WHERE assigned_team LIKE '%Automated%' AND lender_id = ?", (user_id,))
    auto_count = c.fetchone()["cnt"]
    c.execute("SELECT COUNT(*) as cnt FROM institution_cases WHERE assigned_team LIKE '%Legal%' AND lender_id = ?", (user_id,))
    legal_count = c.fetchone()["cnt"]
    conn.close()

    priority_map = {"Low": "low", "Medium": "medium", "High": "high", "Critical": "critical"}
    actions = [
        {
            "id": case["id"],
            "borrower": case["borrower_name"],
            "amount": f"₹{case['outstanding']:,.0f}",
            "type": (
                "Legal" if "Legal" in (case["assigned_team"] or "") else
                "Visit" if "Field" in (case["assigned_team"] or "") else
                "Call"  if "Call" in (case["assigned_team"] or "") else "Email"
            ),
            "due": f"{case['days_overdue']}d overdue" if case["days_overdue"] > 0 else "Current",
            "status": case["status"] if case["status"] in ("resolved", "active") else
                      ("completed" if case["days_overdue"] == 0 else "pending"),
            "priority": priority_map.get(case["priority"], "medium"),
            "priorityLabel": case["priority"],
            "team": case["assigned_team"],
            "action": case["recommended_action"],
            "defaultProb": round((case["default_probability"] or 0) * 100, 1),
        }
        for case in cases
    ]

    return {
        "actions": actions,
        "summary": {
            "completed": f"{completed_count} / {total_count}",
            "completedRatio": int(completed_count / max(total_count, 1) * 100),
            "automated": str(auto_count),
            "automatedRatio": int(auto_count / max(total_count, 1) * 100),
            "legalPending": str(legal_count),
            "legalRatio": int(legal_count / max(total_count, 1) * 100)
        }
    }

@app.patch("/api/recovery/{case_id}")
def update_case_status(case_id: int, body: dict, x_user_id: Optional[str] = Header(None)):
    new_status = body.get("status", "resolved")
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE institution_cases SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
              (new_status, case_id))
    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
              (user_id, "Case Updated", f"Case #{case_id} marked as {new_status}", "institution_cases"))
    conn.commit()
    conn.close()
    return {"ok": True, "case_id": case_id, "status": new_status}

# ============================================================
# INSTITUTION TEAMS (real DB data)
# ============================================================

@app.get("/api/teams")
def get_teams_data(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()
    # Only count cases with status 'active' for team capacity
    c.execute("SELECT assigned_team, COUNT(*) as cnt FROM institution_cases WHERE lender_id = ? AND status = 'active' GROUP BY assigned_team", (user_id,))
    team_counts = {r["assigned_team"]: r["cnt"] for r in c.fetchall()}
    c.execute("SELECT action, details, created_at FROM audit_log WHERE user_id = ? ORDER BY id DESC LIMIT 5", (user_id,))
    audit = c.fetchall()
    conn.close()

    teams = [
        {"id": 1, "name": "Automated System", "iconType": "Cpu",
         "description": "Handles low-risk automated SMS/email reminders.",
         "cases": team_counts.get("Automated System", 0), "capacity": 10000,
         "color": "text-indigo-600", "bg": "bg-indigo-100 dark:bg-indigo-900/30"},
        {"id": 2, "name": "Call Center", "iconType": "Phone",
         "description": "Handles medium-risk calls and standard negotiations.",
         "cases": team_counts.get("Call Center", 0), "capacity": 200,
         "color": "text-blue-600", "bg": "bg-blue-100 dark:bg-blue-900/30"},
        {"id": 3, "name": "Dedicated Field Officers", "iconType": "Users",
         "description": "Handles high-risk field visits and asset verifications.",
         "cases": team_counts.get("Dedicated Field Officers", 0), "capacity": 50,
         "color": "text-emerald-600", "bg": "bg-emerald-100 dark:bg-emerald-900/30"},
        {"id": 4, "name": "Legal Team", "iconType": "ShieldCheck",
         "description": "Handles critical cases requiring legal escalation.",
         "cases": team_counts.get("Legal Team", 0), "capacity": 20,
         "color": "text-rose-600", "bg": "bg-rose-100 dark:bg-rose-900/30"},
    ]

    reassignments = [
        {"id": i+1, "title": log["action"], "desc": log["details"], "time": log["created_at"][:10]}
        for i, log in enumerate(audit)
    ]
    return {"teams": teams, "reassignments": reassignments}

# ============================================================
# HISTORY / AUDIT LOG (real DB data)
# ============================================================

@app.get("/api/history")
def get_history_data(
    search: Optional[str] = None,
    action_type: Optional[str] = None,
    x_user_id: Optional[str] = Header(None)
):
    user_id = get_user_from_token(x_user_id)
    conn = get_conn()
    c = conn.cursor()

    base = """
        SELECT a.id, a.action, a.details, a.created_at, u.full_name as user
        FROM audit_log a
        LEFT JOIN users u ON a.user_id = u.id
        WHERE a.user_id = ?
    """
    params = [user_id]
    clauses = []

    if search:
        clauses.append("(a.action LIKE ? OR a.details LIKE ?)")
        params += [f"%{search}%", f"%{search}%"]
    if action_type and action_type != "all":
        clauses.append("a.action = ?")
        params.append(action_type)

    if clauses:
        base += " AND " + " AND ".join(clauses)
    
    base += " ORDER BY a.id DESC LIMIT 100"
    
    c.execute(base, params)
    logs = c.fetchall()

    # Fetch unique action types for the filter dropdown
    c.execute("SELECT DISTINCT action FROM audit_log WHERE user_id = ?", (user_id,))
    action_types = [r["action"] for r in c.fetchall()]

    conn.close()

    return {
        "logs": [
            {
                "id":      log["id"],
                "date":    log["created_at"][:16],
                "action":  log["action"],
                "details": log["details"],
                "user":    log["user"] or "System"
            }
            for log in logs
        ],
        "total": len(logs),
        "actionTypes": action_types
    }


# ============================================================
# SAVE RISK EVALUATION (borrower)
# ============================================================

class EvaluationSave(BaseModel):
    loan_amount: float
    income: float
    credit_score: int
    property_value: float
    term: int
    employment_status: str
    default_probability: float
    approval_probability: float
    tier: str
    recommended_action: str

@app.post("/api/evaluations/save")
def save_evaluation(data: EvaluationSave, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id) or 1
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO risk_evaluations
        (user_id, loan_amount, income, credit_score, property_value, term,
         employment_status, default_probability, approval_probability, tier, recommended_action)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, data.loan_amount, data.income, data.credit_score, data.property_value,
         data.term, data.employment_status, data.default_probability,
         data.approval_probability, data.tier, data.recommended_action))
    c.execute("""INSERT INTO audit_log (user_id, action, details, entity)
                 VALUES (?,?,?,?)""",
        (user_id, "Risk Simulation", f"Ran pre-approval for ₹{data.loan_amount:,.0f} at {data.tier} tier", "evaluation"))
    conn.commit()
    conn.close()
    return {"status": "saved"}

@app.get("/api/evaluations/history")
def get_evaluation_history(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id) or 1
    conn = get_conn()
    c = conn.cursor()
    c.execute("""SELECT * FROM risk_evaluations WHERE user_id = ? ORDER BY id DESC LIMIT 10""", (user_id,))
    rows = c.fetchall()
    conn.close()
    return {"evaluations": [dict(r) for r in rows]}

# ============================================================
# BORROWER APPLICATIONS
# ============================================================

class BorrowerApplicationInput(BaseModel):
    loan_purpose: str
    loan_amount: float
    term_months: int
    property_value: float = 0.0

@app.post("/api/borrower-applications")
def create_borrower_application(data: BorrowerApplicationInput, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id) or 1
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO borrower_applications (user_id, loan_purpose, loan_amount, term_months, property_value)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, data.loan_purpose, data.loan_amount, data.term_months, data.property_value))
    conn.commit()
    conn.close()
    return {"ok": True, "message": "Application submitted successfully"}

@app.get("/api/borrower-applications")
def get_borrower_applications(x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id) or 1
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM borrower_applications WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return {"applications": [dict(r) for r in rows]}

class AppStatusUpdate(BaseModel):
    status: str

@app.patch("/api/borrower-applications/{app_id}")
def update_application_status(app_id: int, data: AppStatusUpdate, x_user_id: Optional[str] = Header(None)):
    user_id = get_user_from_token(x_user_id) or 1
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM borrower_applications WHERE id = ? AND user_id = ?", (app_id, user_id))
    app_data = c.fetchone()
    if not app_data:
        conn.close()
        raise HTTPException(status_code=404, detail="Application not found.")
        
    old_status = app_data["status"]
    new_status = data.status.lower()
    
    c.execute("UPDATE borrower_applications SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
              (new_status, app_id))
              
    loan_name = f"Application #{app_id} - {app_data['loan_purpose']}"
    
    # Flow: Only ACTIVE/APPROVED applications live in the actual tracked financial overview
    if new_status in ["active", "approved"]:
        # Delete existing tracking just in case to avoid duplicates
        c.execute("DELETE FROM borrower_loans WHERE user_id = ? AND loan_name = ?", (user_id, loan_name))
        
        # Add to active loans
        c.execute("""
            INSERT INTO borrower_loans (user_id, loan_name, loan_amount, outstanding, rate, term_months, status, status_type)
            VALUES (?, ?, ?, ?, ?, ?, 'current', 'good')
        """, (user_id, loan_name, app_data["loan_amount"], app_data["loan_amount"], 9.5, app_data["term_months"]))
    else:
        # Remove from active loans if downgraded back to pending or rejected
        loan_name_prefix = f"Application #{app_id} - %"
        c.execute("DELETE FROM borrower_loans WHERE user_id = ? AND loan_name LIKE ?", (user_id, loan_name_prefix))

    # Recalculate total structural debt
    c.execute("SELECT SUM(outstanding) as total FROM borrower_loans WHERE user_id=?", (user_id,))
    total = c.fetchone()["total"] or 0
    c.execute("UPDATE borrower_profiles SET total_debt=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", (total, user_id))

    conn.commit()
    conn.close()
    return {"ok": True, "status": new_status}

# ============================================================
# INSTITUTION CASES (borrower input)
# ============================================================

@app.get("/api/cases")
def get_cases(x_user_id: Optional[str] = Header(None)):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM institution_cases ORDER BY default_probability DESC")
    cases = c.fetchall()
    conn.close()
    return {"cases": [dict(c) for c in cases]}

class CaseUpdate(BaseModel):
    assigned_team: Optional[str] = None
    status: Optional[str] = None
    recommended_action: Optional[str] = None

@app.patch("/api/cases/{account_id}")
def update_case(account_id: str, data: CaseUpdate, x_user_id: Optional[str] = Header(None)):
    conn = get_conn()
    c = conn.cursor()
    updates = {k: v for k, v in data.dict().items() if v is not None}
    if not updates:
        conn.close()
        return {"status": "no changes"}
    set_clause = ", ".join(f"{k}=?" for k in updates)
    c.execute(f"UPDATE institution_cases SET {set_clause}, updated_at=datetime('now') WHERE account_id=?",
              list(updates.values()) + [account_id])
    user_id = get_user_from_token(x_user_id) or 2
    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
              (user_id, "Case Updated", f"{account_id}: {updates}", "case"))
    conn.commit()
    conn.close()
    return {"status": "updated", "account_id": account_id}

