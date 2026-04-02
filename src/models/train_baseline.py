from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.train_test_split import prepare_data


def train_models():

    X_train, X_test, y_train, y_test = prepare_data()

    # ================================
    # 🔹 Separate columns
    # ================================
    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns

    # ================================
    # 🔹 Preprocessing
    # ================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    # ================================
    # 🔹 Logistic Regression
    # ================================
    log_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    log_model.fit(X_train, y_train)

    y_pred_log = log_model.predict_proba(X_test)[:, 1]
    auc_log = roc_auc_score(y_test, y_pred_log)

    print("\n📊 Logistic Regression AUC:", round(auc_log, 4))


    # ================================
    # 🔹 Decision Tree
    # ================================
    tree_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=10))
    ])

    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict_proba(X_test)[:, 1]
    auc_tree = roc_auc_score(y_test, y_pred_tree)

    print("📊 Decision Tree AUC:", round(auc_tree, 4))


if __name__ == "__main__":
    train_models()