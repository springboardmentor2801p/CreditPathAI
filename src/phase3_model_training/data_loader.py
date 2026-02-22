# ── MUST be first: suppress pkg_resources/setuptools warnings before mlflow loads ──
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Distutils.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Setuptools.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Project root resolution (works regardless of CWD) ────────────────────────
_FILE_DIR     = Path(__file__).resolve().parent          # .../src/phase3_model_training
_PROJECT_ROOT = _FILE_DIR.parent.parent                  # .../CreditPathAI
DATA_RAW      = _PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = _PROJECT_ROOT / "data" / "processed"

# Common target column names for credit default datasets
_TARGET_CANDIDATES = [
    "default", "Default", "DEFAULT",
    "loan_status", "Loan_Status", "LOAN_STATUS",
    "TARGET", "target",
    "defaulted", "Defaulted",
    "default.payment.next.month",
    "bad_flag", "BAD", "bad",
    "y", "label", "Label",
]


def _detect_target(df: pd.DataFrame) -> str:
    for name in _TARGET_CANDIDATES:
        if name in df.columns:
            return name
    last_col = df.columns[-1]
    logger.warning(f"Target column auto-assumed as last column: '{last_col}'")
    return last_col


def _preprocess_raw(df: pd.DataFrame, target_col: str):
    """
    Clean and encode raw credit default data.
    Returns (X: np.ndarray, y: np.ndarray, feature_names: list)
    """
    # Drop unnamed index / ID columns
    drop_cols = [
        c for c in df.columns
        if "unnamed" in c.lower() or c.lower() in ("id", "sk_id_curr", "index")
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped ID/index columns: {drop_cols}")

    # Drop columns with >60% missing values
    missing_frac = df.isnull().mean()
    high_missing = missing_frac[missing_frac > 0.6].index.tolist()
    if high_missing:
        logger.info(f"Dropping {len(high_missing)} columns with >60% missing")
        df = df.drop(columns=high_missing)

    y = df[target_col].copy().astype(int)
    X = df.drop(columns=[target_col]).copy()

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Fill missing values
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        mode_val = X[col].mode()
        X[col] = X[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")

    # Encode categorical columns
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        logger.info(f"One-hot encoded {len(cat_cols)} categorical columns")

    X = X.astype(float)
    feature_names = X.columns.tolist()

    logger.info(f"Preprocessing complete: {X.shape[0]} rows x {X.shape[1]} features")
    logger.info(f"Default rate: {y.mean() * 100:.2f}%")
    return X.values, y.values, feature_names


def load_processed_data():
    """
    Load or generate train/val data splits.

    Priority order:
    1. data/processed/train.csv + val.csv          (already split)
    2. data/processed/features_training.csv         (Phase 1 output — split here)
    3. Any *.csv found in data/raw/                (raw data — preprocess + split)
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    train_csv = DATA_PROCESSED / "train.csv"
    val_csv   = DATA_PROCESSED / "val.csv"

    # ── Priority 1: pre-split processed files ─────────────────────────────────
    if train_csv.exists() and val_csv.exists():
        logger.info(f"Loading pre-split data from: {DATA_PROCESSED}")
        t_df = pd.read_csv(train_csv)
        v_df = pd.read_csv(val_csv)

        target        = _detect_target(t_df)
        feature_names = [c for c in t_df.columns if c != target]

        X_train = t_df[feature_names].values.astype(float)
        y_train = t_df[target].values.astype(int)
        X_val   = v_df[feature_names].values.astype(float)
        y_val   = v_df[target].values.astype(int)

        logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")
        logger.info(
            f"Default rate -> train: {y_train.mean()*100:.2f}% "
            f"| val: {y_val.mean()*100:.2f}%"
        )
        return X_train, X_val, y_train, y_val, feature_names

    # ── Priority 2: Phase 1 output — features_training.csv ───────────────────
    phase1_csv = DATA_PROCESSED / "features_training.csv"
    if phase1_csv.exists():
        logger.info(f"Found Phase 1 output: {phase1_csv.name}")
        df = pd.read_csv(phase1_csv)
        logger.info(f"Loaded shape: {df.shape}")

        # Drop unnamed index columns added by pandas
        drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logger.info(f"Dropped index columns: {drop_cols}")

        target = _detect_target(df)
        logger.info(f"Target column: '{target}'")

        feature_names = [c for c in df.columns if c != target]

        # Ensure all feature columns are numeric; drop non-numeric stragglers
        non_numeric = [
            c for c in feature_names
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            logger.info(f"Encoding non-numeric columns: {non_numeric}")
            df = pd.get_dummies(df, columns=non_numeric, drop_first=True)
            feature_names = [c for c in df.columns if c != target]

        # Fill any remaining NaNs
        df[feature_names] = df[feature_names].fillna(df[feature_names].median())

        X = df[feature_names].values.astype(float)
        y = df[target].values.astype(int)

        logger.info(
            f"Features: {X.shape[1]} | Samples: {X.shape[0]} "
            f"| Default rate: {y.mean()*100:.2f}%"
        )

        # 80 / 20 stratified split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        logger.info(f"Split -> Train: {X_train.shape} | Val: {X_val.shape}")

        # Persist as train.csv / val.csv so future runs skip this step
        t_df = pd.DataFrame(X_train, columns=feature_names)
        t_df[target] = y_train
        v_df = pd.DataFrame(X_val, columns=feature_names)
        v_df[target] = y_val

        t_df.to_csv(train_csv, index=False)
        v_df.to_csv(val_csv,   index=False)
        logger.info(f"Saved train.csv + val.csv to: {DATA_PROCESSED}")

        return (
            X_train.astype(float),
            X_val.astype(float),
            y_train.astype(int),
            y_val.astype(int),
            feature_names,
        )

    # ── Priority 3: raw CSV in data/raw/ ──────────────────────────────────────
    logger.info("No processed data found. Scanning data/raw/ ...")
    raw_csvs = sorted(DATA_RAW.glob("*.csv"))

    if not raw_csvs:
        sep = "=" * 65
        raise FileNotFoundError(
            f"\n{sep}\n"
            f"No data found in any expected location.\n\n"
            f"Your Phase 1 output should be at:\n"
            f"  {phase1_csv}\n\n"
            f"Or place a raw CSV in:\n"
            f"  {DATA_RAW}\n"
            f"{sep}"
        )

    raw_path = raw_csvs[0]
    logger.info(f"Loading raw data: {raw_path.name}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    target = _detect_target(df)
    logger.info(f"Target column detected: '{target}'")

    X, y, feature_names = _preprocess_raw(df, target)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Split -> Train: {X_train.shape} | Val: {X_val.shape}")

    t_df = pd.DataFrame(X_train, columns=feature_names)
    t_df[target] = y_train
    v_df = pd.DataFrame(X_val, columns=feature_names)
    v_df[target] = y_val

    t_df.to_csv(train_csv, index=False)
    v_df.to_csv(val_csv,   index=False)
    logger.info(f"Processed data saved to: {DATA_PROCESSED}")

    return (
        X_train.astype(float),
        X_val.astype(float),
        y_train.astype(int),
        y_val.astype(int),
        feature_names,
    )

