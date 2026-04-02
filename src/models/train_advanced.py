import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.models.train_test_split import prepare_data


def train_advanced():

    X_train, X_test, y_train, y_test = prepare_data()

    # ================================
    # 🔹 Column split
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
    # 🔹 XGBoost
    # ================================
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_pred_xgb)

    print("\n🚀 XGBoost AUC:", round(auc_xgb, 4))


    # ================================
    # 🔹 LightGBM
    # ================================
    lgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lgb_model.fit(X_train, y_train)

    y_pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
    auc_lgb = roc_auc_score(y_test, y_pred_lgb)

    print("🚀 LightGBM AUC:", round(auc_lgb, 4))


if __name__ == "__main__":
    train_advanced()