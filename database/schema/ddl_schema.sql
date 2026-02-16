-- database/schema/ddl_schema.sql

-- ============================================
-- CREDITPATHAI DATABASE SCHEMA
-- ============================================

-- Create database (run this separately as postgres superuser)
-- CREATE DATABASE creditpathai;

-- Connect to database
-- \c creditpathai;

-- ============================================
-- TABLE: borrower_raw (Training data)
-- ============================================
CREATE TABLE IF NOT EXISTS borrower_raw (
    member_id INTEGER PRIMARY KEY,
    residential_state VARCHAR(2),
    years_employment VARCHAR(20),
    home_ownership VARCHAR(20),
    annual_income INTEGER,
    income_verified INTEGER,
    dti_ratio DECIMAL(6, 2),
    length_credit_history INTEGER,
    num_total_credit_lines INTEGER,
    num_open_credit_lines DECIMAL(6, 2),
    num_open_credit_lines_1year INTEGER,
    revolving_balance INTEGER,
    revolving_utilization_rate DECIMAL(6, 2),
    num_derogatory_rec INTEGER,
    num_delinquency_2years INTEGER,
    num_chargeoff_1year INTEGER,
    num_inquiries_6mon INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- TABLE: loan_raw (Training data)
-- ============================================
CREATE TABLE IF NOT EXISTS loan_raw (
    loan_id INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL,
    loan_date DATE,
    purpose VARCHAR(50),
    is_joint_application SMALLINT,
    loan_amount DECIMAL(12, 2),
    term VARCHAR(20),
    interest_rate DECIMAL(6, 2),
    monthly_payment DECIMAL(10, 2),
    grade VARCHAR(5),
    loan_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES borrower_raw(member_id)
);

-- ============================================
-- TABLE: borrower_prod (Production/New data)
-- ============================================
CREATE TABLE IF NOT EXISTS borrower_prod (
    member_id INTEGER PRIMARY KEY,
    residential_state VARCHAR(2),
    years_employment VARCHAR(20),
    home_ownership VARCHAR(20),
    annual_income INTEGER,
    income_verified INTEGER,
    dti_ratio DECIMAL(6, 2),
    length_credit_history INTEGER,
    num_total_credit_lines INTEGER,
    num_open_credit_lines DECIMAL(6, 2),
    num_open_credit_lines_1year INTEGER,
    revolving_balance INTEGER,
    revolving_utilization_rate DECIMAL(6, 2),
    num_derogatory_rec INTEGER,
    num_delinquency_2years INTEGER,
    num_chargeoff_1year INTEGER,
    num_inquiries_6mon INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- TABLE: loan_prod (Production/New data)
-- ============================================
CREATE TABLE IF NOT EXISTS loan_prod (
    loan_id INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL,
    loan_date DATE,
    purpose VARCHAR(50),
    is_joint_application SMALLINT,
    loan_amount DECIMAL(12, 2),
    term VARCHAR(20),
    interest_rate DECIMAL(6, 2),
    monthly_payment DECIMAL(10, 2),
    grade VARCHAR(5),
    loan_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (member_id) REFERENCES borrower_prod(member_id)
);

-- ============================================
-- TABLE: features_training (Engineered features for ML)
-- ============================================
CREATE TABLE IF NOT EXISTS features_training (
    record_id SERIAL PRIMARY KEY,
    member_id INTEGER,
    loan_id INTEGER,
    
    -- Original features
    residential_state VARCHAR(2),
    years_employment VARCHAR(20),
    home_ownership VARCHAR(20),
    annual_income INTEGER,
    income_verified INTEGER,
    dti_ratio DECIMAL(6, 2),
    length_credit_history INTEGER,
    num_total_credit_lines INTEGER,
    num_open_credit_lines DECIMAL(6, 2),
    revolving_balance INTEGER,
    revolving_utilization_rate DECIMAL(6, 2),
    num_derogatory_rec INTEGER,
    num_delinquency_2years INTEGER,
    num_chargeoff_1year INTEGER,
    
    -- Loan features
    loan_amount DECIMAL(12, 2),
    interest_rate DECIMAL(6, 2),
    monthly_payment DECIMAL(10, 2),
    term VARCHAR(20),
    grade VARCHAR(5),
    
    -- Engineered features
    emi_burden DECIMAL(8, 4),
    credit_stress_index DECIMAL(8, 4),
    delinquency_score INTEGER,
    employment_stability_score INTEGER,
    credit_utilization_bucket VARCHAR(20),
    
    -- Target variable
    is_default INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- TABLE: risk_scores (Model predictions)
-- ============================================
CREATE TABLE IF NOT EXISTS risk_scores (
    score_id SERIAL PRIMARY KEY,
    member_id INTEGER,
    loan_id INTEGER,
    default_probability DECIMAL(6, 4),
    risk_bucket VARCHAR(20),
    recommendation TEXT,
    model_version VARCHAR(20),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- INDEXES for performance
-- ============================================

-- Borrower indexes
CREATE INDEX idx_borrower_raw_member ON borrower_raw(member_id);
CREATE INDEX idx_borrower_prod_member ON borrower_prod(member_id);

-- Loan indexes
CREATE INDEX idx_loan_raw_member ON loan_raw(member_id);
CREATE INDEX idx_loan_raw_status ON loan_raw(loan_status);
CREATE INDEX idx_loan_prod_member ON loan_prod(member_id);

-- Features indexes
CREATE INDEX idx_features_member ON features_training(member_id);
CREATE INDEX idx_features_loan ON features_training(loan_id);
CREATE INDEX idx_features_default ON features_training(is_default);

-- Risk scores indexes
CREATE INDEX idx_risk_member ON risk_scores(member_id);
CREATE INDEX idx_risk_bucket ON risk_scores(risk_bucket);

-- ============================================
-- COMMENTS for documentation
-- ============================================
COMMENT ON TABLE borrower_raw IS 'Historical borrower demographic and credit data';
COMMENT ON TABLE loan_raw IS 'Historical loan data with known outcomes';
COMMENT ON TABLE borrower_prod IS 'New borrowers requiring risk assessment';
COMMENT ON TABLE loan_prod IS 'New loans requiring default prediction';
COMMENT ON TABLE features_training IS 'ML-ready features for model training';
COMMENT ON TABLE risk_scores IS 'Model predictions and recommendations';
