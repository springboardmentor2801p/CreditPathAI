# src/phase3_model_training/model_selector.py

import pickle
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Compare all trained models and persist the best one.
    Selection criterion: highest AUC-ROC on the validation set,
    with CV stability (low std) as tiebreaker.
    """

    def __init__(self, models_dir='models/trained'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────

    def compare_models(self, all_metrics, cv_results):
        """
        Build comparison table and return the name of the best model.
        all_metrics : list of metric dicts (from ModelEvaluator)
        cv_results  : {model_name: (fold_aucs, mean_auc, std_auc)}
        """
        logger.info("\n" + "=" * 65)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 65)

        header = (f"{'Model':<30} {'Val AUC':>8} {'CV AUC':>8} "
                  f"{'Precision':>10} {'Recall':>8} {'F1':>7}")
        logger.info(header)
        logger.info("-" * 65)

        scored = []
        for m in all_metrics:
            name = m['model_name']
            cv_mean = cv_results.get(name, (None, 0.0, 1.0))[1]
            cv_std  = cv_results.get(name, (None, 0.0, 1.0))[2]
            composite = m['auc_roc'] * 0.6 + cv_mean * 0.3 - cv_std * 0.1

            logger.info(
                f"{name:<30} {m['auc_roc']:>8.4f} {cv_mean:>8.4f} "
                f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>7.4f}"
            )
            scored.append({'name': name, 'composite': composite,
                           'auc_roc': m['auc_roc'], 'cv_mean': cv_mean,
                           'cv_std': cv_std})

        logger.info("=" * 65)
        best = max(scored, key=lambda x: x['composite'])
        logger.info(f"🏆 Best model: {best['name']}  "
                    f"(composite score = {best['composite']:.4f})")
        return best['name']

    # ─────────────────────────────────────────────────────

    def select_best_model(self, trained_models, all_metrics, cv_results):
        """
        Returns (best_name, best_model_object).
        trained_models: {model_name: model_object}
        """
        best_name = self.compare_models(all_metrics, cv_results)
        best_model = trained_models[best_name]
        return best_name, best_model

    # ─────────────────────────────────────────────────────

    def save_best_model(self, model, model_name, metrics, cv_results):
        """Save best model artifact + metadata JSON."""
        # Save model pickle
        model_path = self.models_dir / 'best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Gather metadata
        cv_mean = cv_results.get(model_name, (None, 0.0, 0.0))[1]
        cv_std  = cv_results.get(model_name, (None, 0.0, 0.0))[2]
        target_metrics = next(
            (m for m in metrics if m['model_name'] == model_name), {}
        )

        metadata = {
            'best_model_name': model_name,
            'model_file':      'best_model.pkl',
            'val_auc_roc':     target_metrics.get('auc_roc'),
            'val_precision':   target_metrics.get('precision'),
            'val_recall':      target_metrics.get('recall'),
            'val_f1':          target_metrics.get('f1_score'),
            'cv_auc_mean':     round(cv_mean, 4),
            'cv_auc_std':      round(cv_std, 4),
        }

        meta_path = self.models_dir / 'best_model_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Best model saved   : {model_path}")
        logger.info(f"✅ Metadata saved     : {meta_path}")
        return str(model_path)

    # ─────────────────────────────────────────────────────

    def load_best_model(self):
        """Load best model and its metadata."""
        model_path = self.models_dir / 'best_model.pkl'
        meta_path  = self.models_dir / 'best_model_metadata.json'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"✅ Best model loaded: {metadata.get('best_model_name')}")
        logger.info(f"   Val AUC-ROC: {metadata.get('val_auc_roc')}")
        return model, metadata
