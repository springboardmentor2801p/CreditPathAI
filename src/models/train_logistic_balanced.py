from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.train_test_split import prepare_data


def train_logistic_balanced():

    X_train, X_test, y_train, y_test = prepare_data()

    # ================================
    # 🔹 Columns
    # ================================
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns

    # ================================
    # 🔹 Preprocessing
    # ================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

    # ================================
    # 🔹 Balanced Logistic Regression
    # ================================
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    print("\n⚖️ Balanced Logistic Regression AUC:", round(auc, 4))


if __name__ == "__main__":
    train_logistic_balanced()