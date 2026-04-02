
import joblib
import lightgbm as lgb

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

from src.models.train_test_split import prepare_data


def save_model():

    X_train, X_test, y_train, y_test = prepare_data()

    # 🔍 DEBUG TARGET (IMPORTANT)
    print("Target distribution:\n", y_train.value_counts())

    # ✅ Separate columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    num_cols = X_train.select_dtypes(exclude=["object"]).columns

    # ✅ Numeric pipeline (IMPROVED)
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())  # 🔥 improves model learning
    ])

    # ✅ Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ✅ Preprocessor
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # 🔥 IMPROVED LIGHTGBM MODEL
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # ✅ Train
    model.fit(X_train, y_train)

    # ✅ Evaluate (VERY IMPORTANT)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\n📈 AUC Score:", roc_auc_score(y_test, y_prob))

    # ✅ Save
    joblib.dump(model, "creditpath_pipeline.pkl")

    print("\n✅ IMPROVED model saved successfully!")


if __name__ == "__main__":
    save_model()
