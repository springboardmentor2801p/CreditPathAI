# ============================================================
# FULL PREPROCESSING + FEATURE ENGINEERING PIPELINE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

file_path = "Loan_Borrower_Merged.csv"
df = pd.read_csv(file_path)

# ------------------------------------------------------------
# 2. TARGET ENCODING
# ------------------------------------------------------------

df["loanStatus"] = df["loanStatus"].map({"Current": 0, "Default": 1})

# ------------------------------------------------------------
# 3. DATE PROCESSING & CHRONOLOGICAL SPLIT PREPARATION
# ------------------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date")

# ------------------------------------------------------------
# 4. REMOVE IDENTIFIERS
# ------------------------------------------------------------

df = df.drop(columns=["loanId", "memberId"])

# ------------------------------------------------------------
# 5. TERM CONVERSION (e.g., "60 months" -> 60)
# ------------------------------------------------------------

df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
df = df.drop(columns=["term"])

# ------------------------------------------------------------
# 6. EMPLOYMENT YEARS CONVERSION (ORDINAL NUMERIC)
# ------------------------------------------------------------

def convert_employment(emp):
    if pd.isna(emp):
        return np.nan
    
    emp = str(emp).strip()
    
    # Case: "< 1 year"
    if "<" in emp:
        return 0.5
    
    # Case: "10+ years"
    if "+" in emp:
        return float(emp.replace("+ years", "").strip())
    
    # Case: "6-9" range → take midpoint
    if "-" in emp:
        parts = emp.split("-")
        try:
            low = float(parts[0])
            high = float(parts[1])
            return (low + high) / 2
        except:
            return np.nan
    
    # Case: "X years" or "X year"
    if "year" in emp:
        try:
            return float(emp.replace(" years", "").replace(" year", "").strip())
        except:
            return np.nan
    
    # Case: already numeric
    try:
        return float(emp)
    except:
        return np.nan

df["yearsEmployment"] = df["yearsEmployment"].apply(convert_employment)

# ------------------------------------------------------------
# 7. GRADE MAPPING (ORDER-PRESERVING FOR LOGISTIC REGRESSION)
# ------------------------------------------------------------

# Extract letter and number
df["grade_letter"] = df["grade"].str[0]
df["grade_number"] = df["grade"].str[1:].astype(int)

# Ordinal map (risk increases alphabetically)
letter_map = {letter: idx for idx, letter in enumerate(sorted(df["grade_letter"].unique()), start=1)}
df["grade_letter_ordinal"] = df["grade_letter"].map(letter_map)

# Create composite ordinal score
df["grade_score"] = df["grade_letter_ordinal"] + (df["grade_number"] / 10)

# Drop original grade columns
df = df.drop(columns=["grade", "grade_letter", "grade_number", "grade_letter_ordinal"])

# ------------------------------------------------------------
# 8. FEATURE ENGINEERING
# ------------------------------------------------------------

# Loan burden ratios
df["loan_to_income_ratio"] = df["loanAmount"] / (df["annualIncome"] + 1)
df["payment_to_income_ratio"] = df["monthlyPayment"] / (df["annualIncome"] + 1)

# Repayment velocity proxies
df["repayment_velocity"] = df["monthlyPayment"] / (df["loanAmount"] + 1)
df["loan_amortization_rate"] = df["loanAmount"] / (df["term_months"] + 1)

# Credit structure features
df["open_credit_ratio"] = df["numOpenCreditLines"] / (df["numTotalCreditLines"] + 1)
df["recent_credit_velocity"] = df["numOpenCreditLines1Year"] / (df["numOpenCreditLines"] + 1)

# Inquiry intensity
df["inquiry_intensity"] = df["numInquiries6Mon"] / (df["lengthCreditHistory"] + 1)

# Delinquency & derogatory density
df["delinquency_density"] = df["numDelinquency2Years"] / (df["lengthCreditHistory"] + 1)
df["derogatory_density"] = df["numDerogatoryRec"] / (df["numTotalCreditLines"] + 1)

# Estimated credit limit from utilization rate
df["estimated_credit_limit"] = df["revolvingBalance"] / ((df["revolvingUtilizationRate"] / 100) + 1e-5)
df["credit_utilization_recomputed"] = df["revolvingBalance"] / (df["estimated_credit_limit"] + 1)

# ------------------------------------------------------------
# 9. LOG TRANSFORM SKEWED FINANCIAL VARIABLES
# ------------------------------------------------------------

skewed_features = ["loanAmount", "annualIncome", "revolvingBalance"]

for col in skewed_features:
    df[f"log_{col}"] = np.log1p(df[col])

# ------------------------------------------------------------
# 10. DROP DATE BEFORE MODELING
# ------------------------------------------------------------

df = df.drop(columns=["date"])

# ------------------------------------------------------------
# 11. OUTLIER TREATMENT (IQR CAPPING - CONTINUOUS ONLY)
# ------------------------------------------------------------

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("loanStatus")

# Identify continuous columns (more than 10 unique values)
continuous_cols = [col for col in numeric_cols if df[col].nunique() > 10]

for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[col] = np.clip(df[col], lower_bound, upper_bound)


# ------------------------------------------------------------
# 12. MEAN IMPUTATION (NUMERIC ONLY)
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# ------------------------------------------------------------
# 13. ONE-HOT ENCODING (EXCLUDE residentialState)
# ------------------------------------------------------------

# Explicitly drop residentialState before encoding
if "residentialState" in df.columns:
    df = df.drop(columns=["residentialState"])

# Identify categorical columns AFTER dropping residentialState
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# One-hot encode remaining categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# ------------------------------------------------------------
# 14. CHRONOLOGICAL SPLIT (80:20)
# ------------------------------------------------------------

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df.drop("loanStatus", axis=1)
y_train = train_df["loanStatus"]

X_test = test_df.drop("loanStatus", axis=1)
y_test = test_df["loanStatus"]

# SAVE TO CSV Before Scaling
df.to_csv("engineered_unscaled.csv", index=False)

# ------------------------------------------------------------
# 15. STANDARD SCALING (FOR LOGISTIC REGRESSION)
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# DATA READY FOR LOGISTIC REGRESSION
# ------------------------------------------------------------

print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)
print("Preprocessing Complete.")

# ------------------------------------------------------------
# 16. RECONSTRUCT DATAFRAMES AFTER SCALING
# ------------------------------------------------------------

# Convert scaled arrays back to DataFrames
X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)

# Append target column
train_processed = pd.concat([X_train_scaled_df, y_train], axis=1)
test_processed = pd.concat([X_test_scaled_df, y_test], axis=1)

# ------------------------------------------------------------
# 17. SAVE TO CSV
# ------------------------------------------------------------

train_processed.to_csv("train_processed.csv", index=False)
test_processed.to_csv("test_processed.csv", index=False)

print("Processed training and testing datasets saved successfully.")

