import pandas as pd

def load_data():
    df = pd.read_csv("data/raw/Loan_Default.csv")

    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())