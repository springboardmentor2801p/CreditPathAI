# ── Warning suppression MUST come before every other import ──────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Saving into deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Scoring failed.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-finite.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*psutil.*")
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
import pickle

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


# ── Custom AUC scorer (bypasses sklearn's is_classifier() check) ──────────────
def _xgb_auc_scorer(estimator, X, y):
    return roc_auc_score(y, estimator.predict_proba(X)[:, 1])


# ── Logistic Regression ───────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    logger.info("MODEL 1: Training Logistic Regression...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc",
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    logger.info(f"Best LR params: {grid.best_params_}")
    logger.info(f"Best LR CV AUC: {grid.best_score_:.4f}")
    logger.info("Logistic Regression trained")
    return grid.best_estimator_


# ── XGBoost ───────────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train):
    logger.info("MODEL 2: Training XGBoost...")
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / max(pos, 1), 2)
    logger.info(f"scale_pos_weight = {spw}  (neg={neg}, pos={pos})")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        verbosity=0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma":            [0, 0.1, 0.3],
        "min_child_weight": [1, 3, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model, param_dist, cv=cv,
        scoring=_xgb_auc_scorer,
        n_iter=25,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info(f"Best XGB params: {search.best_params_}")
    logger.info(f"Best XGB CV AUC: {search.best_score_:.4f}")
    logger.info("XGBoost trained")
    return search.best_estimator_


from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV

def train_lightgbm(X_train, y_train):
    logger.info("MODEL 3: Training LightGBM...")
    model = lgb.LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        verbose=-1,
        random_state=42,
        n_jobs=-1,          # safe with sklearn 1.5.2 — psutil crash is sklearn 1.6 only
    )

    # RandomizedSearchCV: 25 random combos × 5 folds = 125 fits (vs 2560 with GridSearch)
    param_dist = {
        "n_estimators":      [100, 200, 300, 400],
        "max_depth":         [-1, 4, 6, 8],
        "num_leaves":        [31, 63, 127],
        "learning_rate":     [0.01, 0.05, 0.1],
        "subsample":         [0.7, 0.8, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
        "reg_alpha":         [0, 0.1, 0.5],
        "reg_lambda":        [0.5, 1.0, 2.0],
        "min_child_samples": [10, 20, 50],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        model, param_dist, cv=cv,
        scoring="roc_auc",
        n_iter=25,              # 25 random samples from the distribution
        n_jobs=-1,
        verbose=0,
        random_state=42,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info(f"Best LGB params: {search.best_params_}")
    logger.info(f"Best LGB CV AUC: {search.best_score_:.4f}")
    logger.info("LightGBM trained")
    return search.best_estimator_



# ── K-Fold Cross-Validation ───────────────────────────────────────────────────
def cross_validate_model(model, X, y, n_splits=5, model_name="Model"):
    """
    Manual StratifiedKFold CV using sklearn.base.clone() so the
    original fitted model is never modified.
    Returns (fold_aucs, mean_auc, std_auc).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        fold_model = clone(model)
        fold_model.fit(X[tr_idx], y[tr_idx])
        proba = fold_model.predict_proba(X[val_idx])[:, 1]
        auc = float(roc_auc_score(y[val_idx], proba))
        fold_aucs.append(auc)
        logger.info(f"Fold {fold} AUC: {auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    logger.info(f"CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    return fold_aucs, mean_auc, std_auc


# ── Ensemble (soft-voting) ────────────────────────────────────────────────────
class EnsembleModel:
    """
    Soft-voting ensemble.

    BUG FIX: original code stored `self` inside self.models, which caused
    infinite recursion in predict_proba. This version uses a private
    _base_models dict that explicitly filters out EnsembleModel instances.
    """

    def __init__(self, base_models: dict):
        self._base_models = {
            name: m
            for name, m in base_models.items()
            if not isinstance(m, EnsembleModel)
        }
        if not self._base_models:
            raise ValueError("EnsembleModel needs at least one non-Ensemble base model.")

    def predict_proba(self, X):
        probas = np.column_stack([
            m.predict_proba(X)[:, 1]
            for m in self._base_models.values()   # iterates _base_models only
        ])
        avg = probas.mean(axis=1)
        return np.column_stack([1.0 - avg, avg])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @property
    def models(self):
        return dict(self._base_models)

    def __repr__(self):
        return f"EnsembleModel(base={list(self._base_models.keys())})"


# ── Artifact Saving ───────────────────────────────────────────────────────────
def save_artifacts(models: dict, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(output_dir, f"{name.replace(' ', '')}_model.pkl")
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
        logger.info(f"Saved: {path}")
    logger.info(f"Artifacts saved to {output_dir}")
