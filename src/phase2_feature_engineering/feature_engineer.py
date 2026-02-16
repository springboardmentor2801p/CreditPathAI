# src/phase2_feature_engineering/feature_engineer.py

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Create domain-specific features for credit risk prediction
    """
    
    def __init__(self):
        pass
    
    def create_target_variable(self, df):
        """
        Create binary target variable: is_default
        loan_status == 'Default' -> 1, else -> 0
        """
        logger.info("Creating target variable (is_default)...")
        df['is_default'] = (df['loan_status'] == 'Default').astype(int)
        
        default_count = df['is_default'].sum()
        default_rate = (default_count / len(df)) * 100
        
        logger.info(f"✅ Target variable created:")
        logger.info(f"   Default cases: {default_count} ({default_rate:.2f}%)")
        logger.info(f"   Current cases: {len(df) - default_count} ({100-default_rate:.2f}%)")
        
        return df
    
    def create_emi_burden(self, df):
        """
        EMI Burden = monthly_payment / (annual_income / 12)
        Higher burden = higher risk
        """
        logger.info("Creating EMI Burden feature...")
        
        # Calculate monthly income
        df['monthly_income'] = df['annual_income'] / 12
        
        # Calculate EMI burden (percentage)
        df['emi_burden'] = (df['monthly_payment'] / df['monthly_income']) * 100
        
        # Cap extreme values
        df['emi_burden'] = df['emi_burden'].clip(upper=100)
        
        logger.info(f"✅ EMI Burden created (mean: {df['emi_burden'].mean():.2f}%)")
        
        return df
    
    def create_credit_stress_index(self, df):
        """
        Credit Stress Index = dti_ratio * revolving_utilization_rate
        Measures overall credit pressure
        """
        logger.info("Creating Credit Stress Index...")
        
        # Normalize both metrics to 0-1 scale
        dti_normalized = df['dti_ratio'] / 100
        revolving_normalized = df['revolving_utilization_rate'] / 100
        
        # Calculate stress index (0-100 scale)
        df['credit_stress_index'] = (dti_normalized * revolving_normalized) * 100
        
        logger.info(f"✅ Credit Stress Index created (mean: {df['credit_stress_index'].mean():.2f})")
        
        return df
    
    def create_delinquency_score(self, df):
        """
        Delinquency Score = num_delinquency_2years + num_chargeoff_1year
        Total count of negative credit events
        """
        logger.info("Creating Delinquency Score...")
        
        df['delinquency_score'] = (
            df['num_delinquency_2years'] + 
            df['num_chargeoff_1year']
        )
        
        logger.info(f"✅ Delinquency Score created (mean: {df['delinquency_score'].mean():.2f})")
        
        return df
    
    def create_employment_stability_score(self, df):
        """
        Employment Stability Score based on years_employment
        < 1 year: 1
        1 year: 2
        2-5 years: 3
        6-9 years: 4
        10+ years: 5
        """
        logger.info("Creating Employment Stability Score...")
        
        employment_mapping = {
            '< 1 year': 1,
            '1 year': 2,
            '2-5 years': 3,
            '6-9 years': 4,
            '10+ years': 5
        }
        
        df['employment_stability_score'] = df['years_employment'].map(employment_mapping)
        
        # Handle any unmapped values
        df['employment_stability_score'].fillna(1, inplace=True)
        
        logger.info(f"✅ Employment Stability Score created (mean: {df['employment_stability_score'].mean():.2f})")
        
        return df
    
    def create_credit_utilization_bucket(self, df):
        """
        Credit Utilization Buckets:
        Low: < 30%
        Medium: 30-60%
        High: 60-90%
        Very High: > 90%
        """
        logger.info("Creating Credit Utilization Buckets...")
        
        def categorize_utilization(rate):
            if rate < 30:
                return 'Low'
            elif rate < 60:
                return 'Medium'
            elif rate < 90:
                return 'High'
            else:
                return 'Very High'
        
        df['credit_utilization_bucket'] = df['revolving_utilization_rate'].apply(categorize_utilization)
        
        distribution = df['credit_utilization_bucket'].value_counts()
        logger.info(f"✅ Credit Utilization Buckets created:\n{distribution}")
        
        return df
    
    def create_income_to_loan_ratio(self, df):
        """
        Income to Loan Ratio = annual_income / loan_amount
        Higher ratio = lower risk
        """
        logger.info("Creating Income to Loan Ratio...")
        
        df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
        
        logger.info(f"✅ Income to Loan Ratio created (mean: {df['income_to_loan_ratio'].mean():.2f})")
        
        return df
    
    def create_credit_history_length_bucket(self, df):
        """
        Bucket credit history length:
        New: 0-2 years
        Short: 3-5 years
        Medium: 6-10 years
        Long: 11-20 years
        Very Long: > 20 years
        """
        logger.info("Creating Credit History Length Buckets...")
        
        def categorize_history(years):
            if years <= 2:
                return 'New'
            elif years <= 5:
                return 'Short'
            elif years <= 10:
                return 'Medium'
            elif years <= 20:
                return 'Long'
            else:
                return 'Very Long'
        
        df['credit_history_bucket'] = df['length_credit_history'].apply(categorize_history)
        
        distribution = df['credit_history_bucket'].value_counts()
        logger.info(f"✅ Credit History Buckets created:\n{distribution}")
        
        return df
    
    def create_total_credit_exposure(self, df):
        """
        Total Credit Exposure = revolving_balance + loan_amount
        Total outstanding credit obligations
        """
        logger.info("Creating Total Credit Exposure...")
        
        df['total_credit_exposure'] = df['revolving_balance'] + df['loan_amount']
        
        logger.info(f"✅ Total Credit Exposure created (mean: ${df['total_credit_exposure'].mean():.2f})")
        
        return df
    
    def create_inquiry_intensity(self, df):
        """
        Inquiry Intensity = num_inquiries_6mon
        High inquiries suggest credit shopping/desperation
        """
        logger.info("Creating Inquiry Intensity categories...")
        
        def categorize_inquiries(count):
            if count == 0:
                return 'None'
            elif count <= 2:
                return 'Low'
            elif count <= 5:
                return 'Medium'
            else:
                return 'High'
        
        df['inquiry_intensity'] = df['num_inquiries_6mon'].apply(categorize_inquiries)
        
        distribution = df['inquiry_intensity'].value_counts()
        logger.info(f"✅ Inquiry Intensity created:\n{distribution}")
        
        return df
    
    def create_all_features(self, df):
        """
        Create all engineered features
        """
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Create target variable first
        df = self.create_target_variable(df)
        
        # Create all engineered features
        df = self.create_emi_burden(df)
        df = self.create_credit_stress_index(df)
        df = self.create_delinquency_score(df)
        df = self.create_employment_stability_score(df)
        df = self.create_credit_utilization_bucket(df)
        df = self.create_income_to_loan_ratio(df)
        df = self.create_credit_history_length_bucket(df)
        df = self.create_total_credit_exposure(df)
        df = self.create_inquiry_intensity(df)
        
        logger.info("=" * 60)
        logger.info(f"✅ Feature engineering complete!")
        logger.info(f"   Total features: {len(df.columns)}")
        logger.info("=" * 60)
        
        return df
    
    def select_features_for_training(self, df):
        """
        Select final features for model training
        """
        logger.info("Selecting features for training...")
        
        # Core features to keep
        feature_columns = [
            # Identifiers
            'member_id', 'loan_id',
            
            # Original borrower features
            'residential_state', 'years_employment', 'home_ownership',
            'annual_income', 'income_verified', 'dti_ratio',
            'length_credit_history', 'num_total_credit_lines',
            'num_open_credit_lines', 'revolving_balance',
            'revolving_utilization_rate', 'num_derogatory_rec',
            'num_delinquency_2years', 'num_chargeoff_1year',
            'num_inquiries_6mon',
            
            # Loan features
            'loan_amount', 'term', 'interest_rate',
            'monthly_payment', 'grade', 'purpose',
            'is_joint_application',
            
            # Engineered features
            'emi_burden', 'credit_stress_index', 'delinquency_score',
            'employment_stability_score', 'credit_utilization_bucket',
            'income_to_loan_ratio', 'credit_history_bucket',
            'total_credit_exposure', 'inquiry_intensity',
            
            # Target variable
            'is_default'
        ]
        
        # Filter columns that exist in dataframe
        available_columns = [col for col in feature_columns if col in df.columns]
        df_features = df[available_columns].copy()
        
        logger.info(f"✅ Selected {len(available_columns)} features for training")
        
        return df_features
