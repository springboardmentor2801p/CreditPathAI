# src/phase3_model_training/__init__.py
# Updated to match the new functional architecture (no ModelTrainer class)

from .train import (
    train_logistic_regression,
    train_xgboost,
    train_lightgbm,
    cross_validate_model,
    EnsembleModel,
    save_artifacts,
)

from .evaluate import (
    evaluate_model,
    find_best_f1_threshold,
    generate_roc_pr_curves,
    generate_feature_importance_plots,
    generate_cv_fold_plots,
    generate_model_comparison_bar,
    generate_evaluation_report,
    log_model_to_mlflow,
)

from .data_loader import load_processed_data

__all__ = [
    "train_logistic_regression",
    "train_xgboost",
    "train_lightgbm",
    "cross_validate_model",
    "EnsembleModel",
    "save_artifacts",
    "evaluate_model",
    "find_best_f1_threshold",
    "generate_roc_pr_curves",
    "generate_feature_importance_plots",
    "generate_cv_fold_plots",
    "generate_model_comparison_bar",
    "generate_evaluation_report",
    "log_model_to_mlflow",
    "load_processed_data",
]
