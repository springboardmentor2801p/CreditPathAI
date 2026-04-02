from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.train_test_split import prepare_data


def train_balanced():

    X_train, X_test, y_train, y_test = prepare_data()

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    # ================================
    # 🔹 Balanced Decision Tree
    # ================================
    tree_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced"
        ))
    ])

    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    print("\n⚖️ Balanced Decision Tree AUC:", round(auc, 4))


if __name__ == "__main__":
    train_balanced()