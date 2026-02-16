# src/phase2_feature_engineering/feature_store.py

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Store engineered features in PostgreSQL and CSV
    """
    
    def __init__(self, engine):
        self.engine = engine
    
    def save_to_database(self, df, table_name='features_training'):
        """
        Save features to PostgreSQL
        """
        logger.info(f"Saving features to database table: {table_name}...")
        
        try:
            # Drop created_at columns if they exist
            cols_to_drop = ['created_at_x', 'created_at_y', 'created_at', 'loan_date', 'monthly_income']
            df_save = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
            
            # Save to database
            df_save.to_sql(
                table_name,
                self.engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"✅ Saved {len(df_save)} records to {table_name}")
            logger.info(f"   Features: {len(df_save.columns)} columns")
            
        except Exception as e:
            logger.error(f"❌ Failed to save to database: {e}")
            raise
    
    def save_to_csv(self, df, output_path='data/processed/features_training.csv'):
        """
        Save features to CSV for backup/analysis
        """
        logger.info(f"Saving features to CSV: {output_path}...")
        
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Drop timestamp columns
            cols_to_drop = ['created_at_x', 'created_at_y', 'created_at', 'loan_date', 'monthly_income']
            df_save = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
            
            # Save to CSV
            df_save.to_csv(output_path, index=False)
            
            logger.info(f"✅ Saved features to {output_path}")
            logger.info(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Failed to save to CSV: {e}")
            raise
    
    def load_from_database(self, table_name='features_training'):
        """
        Load features from PostgreSQL
        """
        logger.info(f"Loading features from database table: {table_name}...")
        
        try:
            query = f"SELECT * FROM {table_name};"
            df = pd.read_sql(query, self.engine)
            
            logger.info(f"✅ Loaded {len(df)} records from {table_name}")
            logger.info(f"   Features: {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load from database: {e}")
            raise
    
    def load_from_csv(self, input_path='data/processed/features_training.csv'):
        """
        Load features from CSV
        """
        logger.info(f"Loading features from CSV: {input_path}...")
        
        try:
            df = pd.read_csv(input_path)
            
            logger.info(f"✅ Loaded {len(df)} records from CSV")
            logger.info(f"   Features: {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load from CSV: {e}")
            raise
    
    def get_feature_summary(self, df):
        """
        Generate feature summary statistics
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE SUMMARY")
        logger.info("=" * 60)
        
        # Basic stats
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Total features: {len(df.columns)}")
        
        # Target distribution
        if 'is_default' in df.columns:
            default_rate = df['is_default'].mean() * 100
            logger.info(f"\nTarget Distribution:")
            logger.info(f"  Default rate: {default_rate:.2f}%")
            logger.info(f"  Defaults: {df['is_default'].sum()}")
            logger.info(f"  Non-defaults: {(1 - df['is_default']).sum()}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"\nMissing values:")
            logger.info(missing[missing > 0])
        else:
            logger.info(f"\n✅ No missing values!")
        
        # Data types
        logger.info(f"\nData types:")
        logger.info(df.dtypes.value_counts())
        
        logger.info("=" * 60)
