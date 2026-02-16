# src/phase1_data_ingestion/main.py

import sys
sys.path.append('.')

from src.phase1_data_ingestion.data_loader import DataLoader
from src.phase1_data_ingestion.data_cleaner import DataCleaner
from src.phase1_data_ingestion.db_ingestion import DatabaseIngestion
from database.config.db_config import db_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Phase 1: Data Consolidation & Storage - Main Execution
    """
    
    logger.info("=" * 80)
    logger.info("CREDITPATHAI - PHASE 1: DATA CONSOLIDATION & STORAGE")
    logger.info("=" * 80)
    
    # Step 1: Test database connection
    logger.info("\n[STEP 1] Testing database connection...")
    if not db_config.test_connection():
        logger.error("Database connection failed. Please check your configuration.")
        return
    
    # Step 2: Load raw data
    logger.info("\n[STEP 2] Loading raw datasets...")
    loader = DataLoader(data_dir='data/raw')
    datasets = loader.load_all()
    
    # Step 3: Clean data
    logger.info("\n[STEP 3] Cleaning datasets...")
    cleaner = DataCleaner()
    
    borrower_clean = cleaner.clean_borrower(datasets['borrower'])
    loan_clean = cleaner.clean_loan(datasets['loan'])
    borrower_prod_clean, loan_prod_clean = cleaner.clean_production_data(
        datasets['borrower_prod'], 
        datasets['loan_prod']
    )
    
    # Step 4: Ingest into PostgreSQL
    logger.info("\n[STEP 4] Ingesting data into PostgreSQL...")
    ingestion = DatabaseIngestion()
    
    ingestion.ingest_borrower_raw(borrower_clean)
    ingestion.ingest_loan_raw(loan_clean)
    ingestion.ingest_borrower_prod(borrower_prod_clean)
    ingestion.ingest_loan_prod(loan_prod_clean)
    
    # Step 5: Verify ingestion
    logger.info("\n[STEP 5] Verifying data ingestion...")
    ingestion.verify_ingestion()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext Steps:")
    logger.info("  1. Review data in PostgreSQL")
    logger.info("  2. Proceed to Phase 2: Feature Engineering")

if __name__ == "__main__":
    main()
