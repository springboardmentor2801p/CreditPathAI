 import pandas as pd

# Load Dataset
df = pd.read_csv("data_for_training.csv")

print("Original Shape:", df.shape)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

for column in df.select_dtypes(include='object').columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

print("\nMissing values handled.")

# Remove Duplicates
df.drop_duplicates(inplace=True)
print("Duplicates removed.")

# Convert Categorical Columns into Numeric
df = pd.get_dummies(df, drop_first=True)

print("\nAfter Preprocessing Shape:", df.shape)

# Save Cleaned Data
df.to_csv("cleaned_training_data.csv", index=False)

print("\nPreprocessing Completed Successfully!")
