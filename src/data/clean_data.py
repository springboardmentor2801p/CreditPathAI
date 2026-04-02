import pandas as pd


def clean_data(df):

    # 1. Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    print("\nAfter column cleaning:")
    print(df.columns)

    # 2. Drop ID column (not useful for ML)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # 3. Check missing values
    print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False).head(10))

    # 4. Handle missing values

    # Numerical columns → fill with median
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical columns → fill with mode
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nAfter handling missing values:")
    print(df.isnull().sum().sum(), "missing values left")

    return df


if __name__ == "__main__":
    from load_data import load_data

    df = load_data()
    df = clean_data(df)

    print("\nCleaned Data Sample:")
    print(df.head())