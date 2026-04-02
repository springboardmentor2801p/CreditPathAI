from sklearn.model_selection import train_test_split

from src.data.db import load_from_db
from src.data.clean_data import clean_data
from src.features.build_features import build_features


def prepare_data():

    # 1. Load → Clean → Feature
    df = load_from_db()
    df = clean_data(df)
    df = build_features(df)

    # 2. Define target
    # status = 1 → default, 0 → no default
    y = df["status"]

    # 3. Drop target + leakage columns
    drop_cols = [
        "status",                 # target
        "interest_rate_spread",   # leakage
        "rate_of_interest",       # leakage
        "upfront_charges"         # leakage
    ]

    X = df.drop(columns=drop_cols)

    print("\nFinal Features Shape:", X.shape)
    print("Target Shape:", y.shape)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()