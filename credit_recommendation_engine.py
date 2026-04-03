import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt  # Removed as unused

# Load your pre-trained LightGBM model
def load_model(path='best_model_m4.pkl'):
    with open(path, 'rb') as f:
        loaded_object = pickle.load(f)
    if isinstance(loaded_object, dict):
        if 'model' in loaded_object:
            return loaded_object['model']
        elif 'best_model' in loaded_object:
            return loaded_object['best_model']
        elif 'estimator' in loaded_object:
            return loaded_object['estimator']
        else:
            first_value = list(loaded_object.values())[0]
            if hasattr(first_value, 'predict_proba'):
                return first_value
            else:
                raise ValueError('Could not find model in dictionary!')
    else:
        return loaded_object

model = load_model()

# Recovery Efficiency Score calculation
def calculate_recovery_efficiency(row):
    effort_weights = {
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Critical': 4
    }
    effort = effort_weights.get(row['priority'], 2)
    potential_recovery = row['expected_loss']
    efficiency_score = potential_recovery / (effort * 10000)
    return round(efficiency_score, 2)

# Unique additional feature: "Urgency Level" based on days overdue

def urgency_level(days_overdue):
    if days_overdue < 30:
        return 'Normal'
    elif days_overdue < 90:
        return 'Urgent'
    else:
        return 'Critical'
def recommend_action(prob):
    if prob < 0.1:
        return 'Send automated reminder'
    elif prob < 0.3:
        return 'Call borrower and discuss repayment plan'
    elif prob < 0.6:
        return 'Assign dedicated recovery agent'
    else:
        return 'Escalate to senior recovery and legal team'

def expected_loss_engine(input_data):
    """
    Basel III-aligned credit risk engine.
    EL = PD × LGD × EAD

    - PD  (Probability of Default): LightGBM model output, calibrated with credit score, LTV, and term.
    - LGD (Loss Given Default): Collateral-adjusted loss rate based on Loan-to-Value ratio.
    - EAD (Exposure at Default): Total outstanding loan amount.

    Priority is driven by PD buckets (not absolute ₹ amounts), which is the standard banking approach.
    Recovery Efficiency is Net ROI = (Recoverable Value - Actual Cost) / Cost.
    Loan Decision gives lenders a clear APPROVE / CONDITIONAL / REJECT recommendation.
    """
    feature_vector = input_data.get('feature_vector')
    if feature_vector is None:
        raise ValueError('feature_vector missing in input_data')

    raw_prob = model.predict_proba(feature_vector)[0][1]

    credit_score = float(input_data.get('credit_score', 650))
    loan_amount  = float(input_data.get('loan_amount', 100000))
    prop_value   = float(input_data.get('property_value', 0))
    term         = float(input_data.get('term', 180))
    days_overdue = int(input_data.get('days_overdue', 0))
    loan_type    = input_data.get('loan_type', 'secured')
    raw_age = input_data.get('age', 35)
    if isinstance(raw_age, str):
        if '-' in raw_age:
            age = float(raw_age.split('-')[0])
        elif '<' in raw_age:
            age = float(raw_age.replace('<', '').strip())
        elif '>' in raw_age:
            age = float(raw_age.replace('>', '').strip())
        else:
            try:
                age = float(raw_age)
            except:
                age = 35.0
    else:
        age = float(raw_age)

    # LTV logic: only for secured/home/auto loans
    if loan_type in ["secured", "home", "auto"] and prop_value > 0:
        ltv = loan_amount / max(prop_value, 1)
    else:
        ltv = 0

    cs_factor   = max(0.0, min(1.0, (850 - credit_score) / 550))   # 300→1.0, 850→0.0
    ltv_factor  = max(0.0, min(1.0, (ltv - 0.70) / 0.50)) if loan_type in ["secured", "home", "auto"] else 0.0
    term_factor = max(0.0, min(1.0, (term - 120) / 240))            # triggers above 10 yr term
    # Age factor: higher risk for very young (<25) or older (>60) borrowers
    if age < 25:
        age_factor = 0.10
    elif age > 60:
        age_factor = 0.10
    else:
        age_factor = 0.0

    pd = (raw_prob * 0.55) + (cs_factor * 0.20) + (ltv_factor * 0.10) + (term_factor * 0.05) + (age_factor * 0.10)
    pd = max(0.01, min(0.99, pd))

    # LGD logic
    if loan_type in ["unsecured", "education"] or prop_value <= 0:
        lgd = 0.85           # Unsecured or education loan — almost total loss on default
    elif ltv < 0.60:
        lgd = 0.15           # Excellent collateral — lender recovers almost everything
    elif ltv < 0.75:
        lgd = 0.25           # Strong collateral
    elif ltv < 0.90:
        lgd = 0.45           # Adequate collateral, partial recovery expected
    elif ltv < 1.10:
        lgd = 0.60           # Weak collateral, loan near/above property value
    else:
        lgd = 0.80           # Over-leveraged — property won't cover the loan

    # EL = PD × LGD × EAD
    ead = loan_amount        # Exposure at Default = full outstanding amount
    expected_loss = pd * lgd * ead

    # Priority buckets (unchanged)
    if pd < 0.05:
        priority     = "Low"
        team         = "Automated System"
        recovery_cost = 0
        recovery_rate = 0.92       # 92% chance of full recovery with a reminder
    elif pd < 0.20:
        priority     = "Medium"
        team         = "Call Center"
        recovery_cost = 500        # ₹500 agent call cost
        recovery_rate = 0.75
    elif pd < 0.50:
        priority     = "High"
        team         = "Dedicated Field Officers"
        recovery_cost = 2000       # ₹2,000 per field visit
        recovery_rate = 0.55
    else:
        priority     = "Critical"
        team         = "Legal Team"
        recovery_cost = 15000      # ₹15,000 legal processing cost
        recovery_rate = 0.35

    recoverable_value = expected_loss * recovery_rate
    net_gain = recoverable_value - recovery_cost
    if recovery_cost > 0:
        recovery_efficiency = round(net_gain / recovery_cost, 2)
    else:
        recovery_efficiency = round(recovery_rate * 100, 2)

    # Approval logic: for unsecured/education loans, ignore LTV, use credit score and PD
    if (
        (loan_type in ["unsecured", "education"] and pd <= 0.08 and credit_score >= 700)
        or (loan_type in ["secured", "home", "auto"] and pd <= 0.08 and credit_score >= 700 and ltv <= 0.85)
    ):
        loan_decision    = "APPROVE"
        decision_label   = "Approve Loan"
        decision_color   = "emerald"
        decision_reason  = (
            f"Low default risk ({pd*100:.1f}% PD). "
            f"Credit profile is strong (score {int(credit_score)}). "
            + (f"Collateral is adequate (LTV {ltv*100:.0f}%). " if loan_type in ["secured", "home", "auto"] else "")
            + "Recommend standard disbursal."
        )
    elif (
        (loan_type in ["unsecured", "education"] and pd <= 0.30 and credit_score >= 600)
        or (loan_type in ["secured", "home", "auto"] and pd <= 0.30 and credit_score >= 600 and ltv <= 1.00)
    ):
        loan_decision    = "CONDITIONAL"
        decision_label   = "Conditional Approval"
        decision_color   = "amber"
        conditions = []
        if pd > 0.15:
            conditions.append("risk-adjusted interest rate (+150–300 bps)")
        if loan_type in ["secured", "home", "auto"] and ltv > 0.85:
            conditions.append("additional collateral or guarantor required")
        if credit_score < 650:
            conditions.append("co-applicant with strong credit profile recommended")
        if not conditions:
            conditions.append("quarterly review and monitoring recommended")
        decision_reason = (
            f"Moderate risk ({pd*100:.1f}% PD). "
            f"Approve subject to: {'; '.join(conditions)}. "
        )
    else:
        loan_decision    = "REJECT"
        decision_label   = "Reject Application"
        decision_color   = "rose"
        reasons = []
        if pd > 0.30:
            reasons.append(f"high default probability ({pd*100:.1f}%)")
        if credit_score < 600:
            reasons.append(f"insufficient credit score ({int(credit_score)})")
        if loan_type in ["secured", "home", "auto"] and ltv > 1.00:
            reasons.append(f"loan exceeds property value (LTV {ltv*100:.0f}%)")
        if loan_type in ["secured", "home", "auto"] and prop_value <= 0:
            reasons.append("no collateral provided for secured loan")
        decision_reason = f"Application fails risk thresholds: {'; '.join(reasons)}."

    action  = recommend_action(pd)
    urgency = urgency_level(days_overdue)

    return {
        "default_probability":      float(pd),
        "lgd":                       float(lgd),
        "expected_loss":             float(expected_loss),
        "recoverable_value":         float(recoverable_value),
        "priority_level":            priority,
        # "team_assignment":           team,  # Removed for loan applications
        # "recommended_action":        action, # Removed for loan applications
        "recovery_efficiency_score": recovery_efficiency,
        "urgency_level":             urgency,
        "loan_decision":             loan_decision,
        "decision_label":            decision_label,
        "decision_color":            decision_color,
        "decision_reason":           decision_reason,
        "ltv_pct":                   round(ltv * 100, 1),
    }

    recommendations = []
    for i, (prob, exp_loss, loan_amt) in enumerate(zip(probabilities, expected_losses, loan_amounts)):
        if exp_loss < 50000:
            decision = {
                "priority": "Low",
                "assigned_team": "Automated System",
                "recovery_channel": "Email + SMS Reminder",
                "follow_up_frequency": "Once in 15 days",
                "legal_action": False,
                "recommended_action": "Send automated reminder"
            }
        elif exp_loss < 200000:
            decision = {
                "priority": "Medium",
                "assigned_team": "Call Center Agent",
                "recovery_channel": "Phone Call + EMI Restructure Offer",
                "follow_up_frequency": "Weekly",
                "legal_action": False,
                "recommended_action": "Call borrower and discuss repayment plan"
            }
        elif exp_loss < 500000:
            decision = {
                "priority": "High",
                "assigned_team": "Dedicated Recovery Officer",
                "recovery_channel": "Direct Call + Field Visit",
                "follow_up_frequency": "Every 5 days",
                "legal_action": False,
                "recommended_action": "Assign dedicated recovery agent"
            }
        else:
            decision = {
                "priority": "Critical",
                "assigned_team": "Senior Recovery & Legal Team",
                "recovery_channel": "Legal Notice + Field Investigation",
                "follow_up_frequency": "Every 3 days",
                "legal_action": True,
                "recommended_action": "Escalate to senior recovery and legal team"
            }
        rec = {
            'borrower_index': i,
            'loan_amount': float(loan_amt),
            'default_probability': float(prob),
            'expected_loss': float(exp_loss),
            'priority': decision['priority'],
            'assigned_team': decision['assigned_team'],
            'recovery_channel': decision['recovery_channel'],
            'follow_up_frequency': decision['follow_up_frequency'],
            'legal_action': decision['legal_action'],
            'recommended_action': decision['recommended_action']
        }
        if borrower_info is not None and i < len(borrower_info):
            for col in borrower_info.columns:
                if col not in rec:
                    rec[col] = borrower_info.iloc[i][col]
        rec['recovery_efficiency_score'] = calculate_recovery_efficiency(rec)
        rec['urgency_level'] = urgency_level(rec.get('days_overdue', 0))
        recommendations.append(rec)
    df = pd.DataFrame(recommendations)
    return df

# Example usage (replace with your actual data loading)
if __name__ == "__main__":
    # Sample data
    num_borrowers = 5
    num_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 20
    X_sample = np.random.randn(num_borrowers, num_features)
    loan_amounts_sample = np.array([30000, 150000, 400000, 1000000, 2500000])
    borrower_info = pd.DataFrame({
        'account_id': [f'ACC{1000+i}' for i in range(num_borrowers)],
        'borrower_name': [f'Borrower {i+1}' for i in range(num_borrowers)],
        'days_overdue': np.random.randint(0, 180, num_borrowers),
        'credit_score': np.random.randint(300, 850, num_borrowers)
    })
    df = batch_expected_loss_recommendations(X_sample, loan_amounts_sample, borrower_info, model)
    print(df[['account_id','loan_amount','default_probability','expected_loss','priority','recovery_efficiency_score','urgency_level','recommended_action']].to_string(index=False))
