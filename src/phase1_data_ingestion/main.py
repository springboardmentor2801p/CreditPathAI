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
    
    try:
        logger.info("=" * 80)
        logger.info("CREDITPATHAI - PHASE 1: DATA CONSOLIDATION & STORAGE")
        logger.info("=" * 80)
        
        # Step 1: Test database connection
        logger.info("\n[STEP 1] Testing database connection...")
        if not db_config.test_connection():
            logger.error("❌ Database connection failed. Please check your configuration.")
            return False
        
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
        
        logger.info("\nData cleaning summary:")
        logger.info(f"  • Training borrowers: {len(borrower_clean):,}")
        logger.info(f"  • Training loans:     {len(loan_clean):,}")
        logger.info(f"  • Production borrowers: {len(borrower_prod_clean)}")
        logger.info(f"  • Production loans:     {len(loan_prod_clean)}")
        
        # Step 4: Initialize database ingestion
        logger.info("\n[STEP 4] Preparing database...")
        ingestion = DatabaseIngestion()
        ingestion.clear_all_data()
        
        # Step 5: Ingest data in correct order (parent tables first)
        logger.info("\n[STEP 5] Ingesting data into PostgreSQL...")
        logger.info("Note: This may take a few minutes for large datasets...\n")
        
        # Insert borrower tables first (parent tables)
        ingestion.ingest_borrower_raw(borrower_clean)
        ingestion.ingest_borrower_prod(borrower_prod_clean)
        
        # Then insert loan tables (child tables with foreign keys)
        ingestion.ingest_loan_raw(loan_clean)
        ingestion.ingest_loan_prod(loan_prod_clean)
        
        # Step 6: Verify ingestion
        logger.info("\n[STEP 6] Verifying data ingestion...")
        ingestion.verify_ingestion()
        
        # Success summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ PHASE 1 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\n📊 Summary:")
        logger.info(f"  • {len(borrower_clean):,} borrowers loaded")
        logger.info(f"  • {len(loan_clean):,} loans loaded")
        logger.info(f"  • {len(borrower_prod_clean)} production borrowers loaded")
        logger.info(f"  • {len(loan_prod_clean)} production loans loaded")
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Verify data in PostgreSQL:")
        logger.info("     psql -U postgres -d creditpathai")
        logger.info("     \\dt")
        logger.info("     SELECT COUNT(*) FROM borrower_raw;")
        logger.info("  2. Proceed to Phase 2: Feature Engineering")
        logger.info("     python src\\phase2_feature_engineering\\main.py")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ PHASE 1 FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check PostgreSQL is running")
        logger.error("  2. Verify database 'creditpathai' exists")
        logger.error("  3. Check .env file has correct credentials")
        logger.error("  4. Ensure schema is created (run DDL script)")
        logger.error("=" * 80)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
