# src/phase2_feature_engineering/preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Preprocess raw data for ML model training
    """
    
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data_from_db(self, engine):
        """
        Load and merge borrower and loan data from PostgreSQL
        """
        logger.info("Loading data from PostgreSQL...")
        
        # Load borrower data
        query_borrower = "SELECT * FROM borrower_raw;"
        df_borrower = pd.read_sql(query_borrower, engine)
        logger.info(f"Loaded {len(df_borrower)} borrower records")
        
        # Load loan data
        query_loan = "SELECT * FROM loan_raw;"
        df_loan = pd.read_sql(query_loan, engine)
        logger.info(f"Loaded {len(df_loan)} loan records")
        
        # Merge on member_id
        df_merged = pd.merge(
            df_loan,
            df_borrower,
            on='member_id',
            how='inner'
        )
        
        logger.info(f"✅ Merged dataset: {len(df_merged)} records")
        logger.info(f"Columns: {len(df_merged.columns)}")
        
        return df_merged
    
    def handle_missing_values(self, df):
        """
        Handle missing values in numeric and categorical columns
        """
        logger.info("Handling missing values...")
        df_clean = df.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Remove non-feature columns from imputation
        exclude_cols = ['member_id', 'loan_id', 'loan_date', 'created_at_x', 'created_at_y', 'loan_status']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # Log missing values before imputation
        missing_before = df_clean[numeric_cols + categorical_cols].isnull().sum()
        missing_cols = missing_before[missing_before > 0]
        if len(missing_cols) > 0:
            logger.info(f"Missing values detected:\n{missing_cols}")
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = self.numeric_imputer.fit_transform(df_clean[numeric_cols])
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            df_clean[categorical_cols] = self.categorical_imputer.fit_transform(df_clean[categorical_cols])
        
        logger.info(f"✅ Missing values handled for {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
        
        return df_clean
    
    def handle_outliers(self, df, columns=None, method='iqr', threshold=3.0):
        """
        Handle outliers using IQR method or Z-score
        """
        logger.info(f"Handling outliers using {method} method...")
        df_clean = df.copy()
        
        if columns is None:
            # Select numeric columns
            columns = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
            exclude_cols = ['member_id', 'loan_id', 'is_joint_application']
            columns = [col for col in columns if col not in exclude_cols]
        
        outlier_count = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                before = (df_clean[col] < lower_bound).sum() + (df_clean[col] > upper_bound).sum()
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_count += before
                
            elif method == 'zscore':
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                before = (df_clean[col] < lower_bound).sum() + (df_clean[col] > upper_bound).sum()
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_count += before
        
        logger.info(f"✅ Capped {outlier_count} outliers across {len(columns)} columns")
        
        return df_clean
    
    def encode_categorical_features(self, df, categorical_cols=None):
        """
        Encode categorical features using Label Encoding
        """
        logger.info("Encoding categorical features...")
        df_encoded = df.copy()
        
        if categorical_cols is None:
            categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
            exclude_cols = ['loan_date', 'created_at_x', 'created_at_y', 'loan_status']
            categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Handle any remaining NaN values
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"  Encoded {col}: {len(le.classes_)} unique values")
        
        logger.info(f"✅ Encoded {len(categorical_cols)} categorical features")
        
        return df_encoded
    
    def scale_numeric_features(self, df, numeric_cols=None):
        """
        Scale numeric features using StandardScaler
        """
        logger.info("Scaling numeric features...")
        df_scaled = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()
            exclude_cols = ['member_id', 'loan_id', 'is_joint_application', 'income_verified', 
                          'num_derogatory_rec', 'num_delinquency_2years', 'num_chargeoff_1year']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Only scale if columns exist
        cols_to_scale = [col for col in numeric_cols if col in df_scaled.columns]
        
        if len(cols_to_scale) > 0:
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
            logger.info(f"✅ Scaled {len(cols_to_scale)} numeric features")
        
        return df_scaled
    
    def save_preprocessors(self, output_dir='models/trained'):
        """
        Save preprocessing objects for later use
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        with open(output_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoders
        with open(output_path / 'label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save imputers
        with open(output_path / 'numeric_imputer.pkl', 'wb') as f:
            pickle.dump(self.numeric_imputer, f)
        
        with open(output_path / 'categorical_imputer.pkl', 'wb') as f:
            pickle.dump(self.categorical_imputer, f)
        
        logger.info(f"✅ Saved preprocessors to {output_path}")
    
    def load_preprocessors(self, input_dir='models/trained'):
        """
        Load saved preprocessing objects
        """
        input_path = Path(input_dir)
        
        with open(input_path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(input_path / 'label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(input_path / 'numeric_imputer.pkl', 'rb') as f:
            self.numeric_imputer = pickle.load(f)
        
        with open(input_path / 'categorical_imputer.pkl', 'rb') as f:
            self.categorical_imputer = pickle.load(f)
        
        logger.info(f"✅ Loaded preprocessors from {input_path}")
