import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PostgreSQL Connection
# -------------------------------
USERNAME = "postgres"
PASSWORD = "12345"
HOST = "localhost"
PORT = "5432"
DATABASE = "creditpathai"

engine = create_engine(
    f'postgresql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
)

try:
    # -------------------------------
    # Load Feature Engineered Data
    # -------------------------------
    df = pd.read_sql("SELECT * FROM feature_engineered_data", engine)

    print("Data Loaded Successfully!")

    # -------------------------------
    # Separate Features and Target
    # -------------------------------
    X = df.drop("target", axis=1)
    y = df["target"]

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Feature Scaling
    # -------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # Logistic Regression Model
    # -------------------------------
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # -------------------------------
    # Predictions
    # -------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------------
    # Evaluation
    # -------------------------------
    auc_score = roc_auc_score(y_test, y_prob)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Baseline AUC-ROC Score:", auc_score)

except Exception as e:
    print("Error:", e)