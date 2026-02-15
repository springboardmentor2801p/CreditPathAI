import pandas as pd


def generate_kpi_report(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    print("\n========== KPI REPORT ==========\n")

    # Basic KPIs
    total_customers = len(df)
    default_rate = df['target'].mean() * 100
    avg_income = df['monthly_income'].mean()
    avg_expense = df['monthly_expense'].mean()
    avg_wallet_balance = df['avg_wallet_balance'].mean()
    avg_payment_ratio = df['on_time_payment_ratio'].mean()
    avg_loans = df['num_loans_taken'].mean()

    # Print KPIs
    print(f"Total Customers               : {total_customers}")
    print(f"Default Rate (%)              : {default_rate:.2f}%")
    print(f"Average Monthly Income        : {avg_income:.2f}")
    print(f"Average Monthly Expense       : {avg_expense:.2f}")
    print(f"Average Wallet Balance        : {avg_wallet_balance:.2f}")
    print(f"Average On-Time Payment Ratio : {avg_payment_ratio:.2f}")
    print(f"Average Loans Taken           : {avg_loans:.2f}")

    print("\n=================================\n")


if __name__ == "__main__":
    generate_kpi_report("cleaned_training_data.csv")
