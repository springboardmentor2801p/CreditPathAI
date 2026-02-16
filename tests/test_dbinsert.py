# test_insert.py - Create this file in the root directory

import sys
sys.path.append('.')

import pandas as pd
from database.config.db_config import db_config
from src.phase1_data_ingestion.data_loader import DataLoader
from src.phase1_data_ingestion.data_cleaner import DataCleaner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_small_insert():
    """Test with a small sample first"""
    
    logger.info("Testing small data insert...")
    
    # Load and clean data
    loader = DataLoader(data_dir='data/raw')
    datasets = loader.load_all()
    
    cleaner = DataCleaner()
    borrower_clean = cleaner.clean_borrower(datasets['borrower'])
    loan_clean = cleaner.clean_loan(datasets['loan'])
    
    # Take small samples
    borrower_sample = borrower_clean.head(100)
    loan_sample = loan_clean[loan_clean['member_id'].isin(borrower_sample['member_id'])].head(100)
    
    logger.info(f"Sample sizes: {len(borrower_sample)} borrowers, {len(loan_sample)} loans")
    
    # Get engine
    engine = db_config.create_engine()
    
    # Test insert
    try:
        logger.info("Inserting borrower sample...")
        borrower_sample.to_sql('borrower_raw', engine, if_exists='append', index=False, method='multi')
        logger.info("✅ Borrower sample inserted")
        
        logger.info("Inserting loan sample...")
        loan_sample.to_sql('loan_raw', engine, if_exists='append', index=False, method='multi')
        logger.info("✅ Loan sample inserted")
        
        logger.info("\n✅ TEST SUCCESSFUL! Now run the full script.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_small_insert()
