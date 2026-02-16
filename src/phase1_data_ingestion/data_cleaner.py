# src/phase1_data_ingestion/data_cleaner.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Clean and standardize data before database ingestion
    """
    
    @staticmethod
    def standardize_column_names(df):
        """
        Convert column names to snake_case
        """
        df.columns = df.columns.str.replace('([a-z0-9])([A-Z])', r'\1_\2', regex=True)
        df.columns = df.columns.str.lower()
        return df
    
    @staticmethod
    def clean_borrower(df):
        """
        Clean borrower dataset
        """
        logger.info("Cleaning borrower data...")
        df_clean = df.copy()
        
        # Standardize column names
        df_clean = DataCleaner.standardize_column_names(df_clean)
        
        # Handle missing values
        # numOpenCreditLines has 967 missing - fill with median
        if df_clean['num_open_credit_lines'].isnull().sum() > 0:
            median_val = df_clean['num_open_credit_lines'].median()
            df_clean['num_open_credit_lines'].fillna(median_val, inplace=True)
            logger.info(f"Filled {df_clean['num_open_credit_lines'].isnull().sum()} missing values in num_open_credit_lines with median: {median_val}")
        
        # Validate data types
        numeric_cols = [
            'annual_income', 'dti_ratio', 'length_credit_history',
            'num_total_credit_lines', 'num_open_credit_lines',
            'revolving_balance', 'revolving_utilization_rate'
        ]
        
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove duplicates based on member_id
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['member_id'], keep='first')
        removed = initial_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate member_id records")
        
        logger.info(f"✅ Borrower data cleaned: {len(df_clean)} records")
        return df_clean
    
    @staticmethod
    def clean_loan(df):
        """
        Clean loan dataset
        """
        logger.info("Cleaning loan data...")
        df_clean = df.copy()
        
        # Standardize column names
        df_clean = DataCleaner.standardize_column_names(df_clean)
        
        # Parse date column
        df_clean['loan_date'] = pd.to_datetime(df_clean['date'], format='%m/%d/%Y', errors='coerce')
        df_clean = df_clean.drop('date', axis=1)
        
        # Handle missing values
        # isJointApplication - fill with 0 (assuming not joint)
        df_clean['is_joint_application'].fillna(0, inplace=True)
        
        # loanAmount - drop rows with missing (1006 records)
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['loan_amount'])
        logger.info(f"Dropped {initial_count - len(df_clean)} rows with missing loan_amount")
        
        # term - fill missing with mode
        if df_clean['term'].isnull().sum() > 0:
            mode_term = df_clean['term'].mode()[0]
            df_clean['term'].fillna(mode_term, inplace=True)
            logger.info(f"Filled missing term with mode: {mode_term}")
        
        # monthlyPayment - calculate from loanAmount and interestRate if missing
        # For simplicity, we'll keep as is since there are no missing values
        
        # Validate loan_status
        valid_statuses = ['Current', 'Default']
        df_clean = df_clean[df_clean['loan_status'].isin(valid_statuses)]
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['loan_id'], keep='first')
        removed = initial_count - len(df_clean)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate loan_id records")
        
        logger.info(f"✅ Loan data cleaned: {len(df_clean)} records")
        return df_clean
    
    @staticmethod
    def clean_production_data(df_borrower, df_loan):
        """
        Clean production datasets (similar logic)
        """
        logger.info("Cleaning production data...")
        
        # Clean borrower
        df_borrower_clean = DataCleaner.standardize_column_names(df_borrower.copy())
        df_borrower_clean['num_open_credit_lines'].fillna(
            df_borrower_clean['num_open_credit_lines'].median(), inplace=True
        )
        
        # Clean loan
        df_loan_clean = DataCleaner.standardize_column_names(df_loan.copy())
        df_loan_clean['loan_date'] = pd.to_datetime(df_loan_clean['date'], format='%m/%d/%Y', errors='coerce')
        df_loan_clean = df_loan_clean.drop('date', axis=1)
        df_loan_clean['is_joint_application'].fillna(0, inplace=True)
        
        logger.info(f"✅ Production data cleaned: {len(df_borrower_clean)} borrowers, {len(df_loan_clean)} loans")
        return df_borrower_clean, df_loan_clean
