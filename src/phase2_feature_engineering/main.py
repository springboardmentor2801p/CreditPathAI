# src/phase2_feature_engineering/main.py

import sys
sys.path.append('.')

from src.phase2_feature_engineering.preprocessor import DataPreprocessor
from src.phase2_feature_engineering.feature_engineer import FeatureEngineer
from src.phase2_feature_engineering.feature_store import FeatureStore
from database.config.db_config import db_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Phase 2: Data Preprocessing & Feature Engineering - Main Execution
    """
    
    logger.info("=" * 80)
    logger.info("CREDITPATHAI - PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Step 1: Initialize components
    logger.info("\n[STEP 1] Initializing components...")
    engine = db_config.create_engine()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    feature_store = FeatureStore(engine)
    
    # Step 2: Load data from PostgreSQL
    logger.info("\n[STEP 2] Loading data from PostgreSQL...")
    df_raw = preprocessor.load_data_from_db(engine)
    logger.info(f"Initial dataset shape: {df_raw.shape}")
    
    # Step 3: Data Preprocessing
    logger.info("\n[STEP 3] Data Preprocessing...")
    
    # 3.1: Handle missing values
    df_preprocessed = preprocessor.handle_missing_values(df_raw)
    
    # 3.2: Handle outliers
    df_preprocessed = preprocessor.handle_outliers(df_preprocessed, method='iqr')
    
    # Step 4: Feature Engineering
    logger.info("\n[STEP 4] Feature Engineering...")
    df_features = feature_engineer.create_all_features(df_preprocessed)
    
    # Step 5: Select final features
    logger.info("\n[STEP 5] Selecting features for training...")
    df_final = feature_engineer.select_features_for_training(df_features)
    
    # Step 6: Encode categorical features (for storage)
    logger.info("\n[STEP 6] Final preprocessing...")
    # Note: We keep categorical as-is for now, will encode during model training
    
    # Step 7: Store features
    logger.info("\n[STEP 7] Storing engineered features...")
    feature_store.save_to_database(df_final, table_name='features_training')
    feature_store.save_to_csv(df_final, output_path='data/processed/features_training.csv')
    
    # Step 8: Feature summary
    logger.info("\n[STEP 8] Generating feature summary...")
    feature_store.get_feature_summary(df_final)
    
    # Step 9: Save preprocessors
    logger.info("\n[STEP 9] Saving preprocessing artifacts...")
    preprocessor.save_preprocessors(output_dir='models/trained')
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 2 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nDeliverables:")
    logger.info("  ✅ Preprocessed data with missing values handled")
    logger.info("  ✅ Outliers capped using IQR method")
    logger.info("  ✅ 9 engineered features created:")
    logger.info("     - EMI Burden")
    logger.info("     - Credit Stress Index")
    logger.info("     - Delinquency Score")
    logger.info("     - Employment Stability Score")
    logger.info("     - Credit Utilization Bucket")
    logger.info("     - Income to Loan Ratio")
    logger.info("     - Credit History Bucket")
    logger.info("     - Total Credit Exposure")
    logger.info("     - Inquiry Intensity")
    logger.info("  ✅ Features stored in PostgreSQL (features_training table)")
    logger.info("  ✅ Features saved to CSV (data/processed/features_training.csv)")
    logger.info("  ✅ Preprocessing artifacts saved (models/trained/)")
    logger.info("\nNext Steps:")
    logger.info("  1. Review engineered features")
    logger.info("  2. Proceed to Phase 3: Model Training & Evaluation")

if __name__ == "__main__":
    main()
