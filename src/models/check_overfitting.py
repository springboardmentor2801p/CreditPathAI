import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.train_test_split import prepare_data


def check_model():

    X_train, X_test, y_train, y_test = prepare_data()

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "string"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # Train prediction
    y_train_pred = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)

    # Test prediction
    y_test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)

    print("\n📊 Train AUC:", round(train_auc, 4))
    print("📊 Test AUC :", round(test_auc, 4))

    gap = train_auc - test_auc

    print("\nAUC Gap:", round(gap, 4))

    if gap < 0.03:
        print("✅ Model is well-generalized (no overfitting)")
    elif gap < 0.1:
        print("⚠️ Mild overfitting")
    else:
        print("❌ High overfitting")


if __name__ == "__main__":
    check_model()