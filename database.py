"""
CreditPath AI - Database Layer
Creates app.db with users, borrower profiles, loans, evaluations, institution cases tables.
Run this once to seed the database.
"""
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
import random

DB_PATH = "app.db"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    # ---- USERS TABLE ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('borrower', 'institution')),
        full_name TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---- BORROWER PROFILES ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS borrower_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),
        credit_score INTEGER DEFAULT 700,
        annual_income REAL DEFAULT 500000,
        total_debt REAL DEFAULT 0,
        debt_to_income REAL DEFAULT 0,
        next_payment_due REAL DEFAULT 0,
        next_payment_days INTEGER DEFAULT 30,
        employment_status TEXT DEFAULT 'employed',
        credit_score_history TEXT DEFAULT '[]',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---- BORROWER LOANS ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS borrower_loans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        loan_name TEXT NOT NULL,
        loan_amount REAL NOT NULL,
        outstanding REAL NOT NULL,
        rate REAL NOT NULL,
        term_months INTEGER NOT NULL,
        months_paid INTEGER DEFAULT 0,
        status TEXT DEFAULT 'current',
        status_type TEXT DEFAULT 'good',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---- RISK EVALUATIONS (borrower sim history) ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS risk_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        loan_amount REAL,
        income REAL,
        credit_score INTEGER,
        property_value REAL,
        term INTEGER,
        employment_status TEXT,
        default_probability REAL,
        approval_probability REAL,
        tier TEXT,
        recommended_action TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---- INSTITUTION CASES (lender borrower registry) ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS institution_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id TEXT UNIQUE NOT NULL,
        lender_id INTEGER REFERENCES users(id),
        borrower_name TEXT NOT NULL,
        loan_amount REAL NOT NULL,
        outstanding REAL NOT NULL,
        credit_score INTEGER,
        days_overdue INTEGER DEFAULT 0,
        default_probability REAL,
        priority TEXT DEFAULT 'Low',
        assigned_team TEXT DEFAULT 'Automated System',
        recommended_action TEXT,
        status TEXT DEFAULT 'active',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---- AUDIT LOG ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(id),
        action TEXT NOT NULL,
        details TEXT,
        entity TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ---- BORROWER APPLICATIONS ----
    c.execute("""
    CREATE TABLE IF NOT EXISTS borrower_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL REFERENCES users(id),
        loan_purpose TEXT,
        loan_amount REAL,
        term_months INTEGER,
        property_value REAL,
        status TEXT DEFAULT 'pending',
        assigned_lender_id INTEGER REFERENCES users(id),
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()

    # ---- SEED DATA ----
    seed_data(conn)
    conn.close()
    print("Database initialised successfully.")

def seed_data(conn):
    c = conn.cursor()

    # Check if already seeded
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] > 0:
        print("Database already seeded.")
        return

    # --- Seed Borrower User ---
    c.execute("""INSERT INTO users (email, password_hash, role, full_name) VALUES (?,?,?,?)""",
        ("borrower@creditpath.ai", hash_password("borrower123"), "borrower", "Arjun Sharma"))
    borrower_id = c.lastrowid

    # Credit score history for the past 12 months
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    scores = [680, 685, 682, 690, 705, 710, 708, 715, 718, 720, 722, 724]
    score_history = json.dumps({"labels": months, "values": scores})

    total_debt = 1542000  # 15.42 Lakh
    annual_income = 1800000  # 18 Lakh
    dti = round((total_debt / annual_income) * 100, 1)

    c.execute("""INSERT INTO borrower_profiles
        (user_id, credit_score, annual_income, total_debt, debt_to_income,
         next_payment_due, next_payment_days, employment_status, credit_score_history)
        VALUES (?,?,?,?,?,?,?,?,?)""",
        (borrower_id, 724, annual_income, total_debt, dti, 45000, 14, "employed", score_history))

    # Seed Borrower Loans
    loans = [
        (borrower_id, "Auto Loan (HDFC)", 850000, 627400, 8.5, 48, 12, "current", "good"),
        (borrower_id, "Credit Card (Axis)", 420000, 420000, 36.0, 0, 0, "high_util", "warning"),
        (borrower_id, "Personal Loan (SBI)", 272000, 215600, 12.0, 36, 8, "current", "good"),
    ]
    c.executemany("""INSERT INTO borrower_loans
        (user_id, loan_name, loan_amount, outstanding, rate, term_months, months_paid, status, status_type)
        VALUES (?,?,?,?,?,?,?,?,?)""", loans)

    # Audit log entries for borrower
    audit_entries = [
        (borrower_id, "Payment Posted", "Applied ₹45,000 to HDFC Auto Loan", "loan"),
        (borrower_id, "Credit Score Update", "Credit score rose to 724 (+15 pts)", "profile"),
        (borrower_id, "Risk Simulation", "Ran pre-approval simulation for ₹50L home loan", "evaluation"),
        (borrower_id, "Information Update", "Updated employment details", "profile"),
        (borrower_id, "Credit Check", "Soft pull by LoanPartner", "profile"),
    ]
    c.executemany("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)", audit_entries)

    # --- Seed Institution User (clean slate — no pre-seeded cases) ---
    c.execute("""INSERT INTO users (email, password_hash, role, full_name) VALUES (?,?,?,?)""",
        ("lender@creditpath.ai", hash_password("lender123"), "institution", "Priya Mehta"))
    lender_id = c.lastrowid

    # One welcome audit entry — no fake cases or fabricated history
    c.execute("INSERT INTO audit_log (user_id, action, details, entity) VALUES (?,?,?,?)",
        (lender_id, "Account Created", "Institution account initialised — portfolio is empty and ready.", "users"))

    conn.commit()
    print("Seed data inserted.")


if __name__ == "__main__":
    init_db()
