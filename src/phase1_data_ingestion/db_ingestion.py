# src/phase1_data_ingestion/db_ingestion.py

import pandas as pd
from sqlalchemy import text
import logging
import sys
sys.path.append('.')

from database.config.db_config import db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseIngestion:
    """
    Ingest cleaned data into PostgreSQL
    """
    
    def __init__(self):
        self.engine = db_config.create_engine()
    
    def ingest_borrower_raw(self, df):
        """
        Insert borrower training data
        """
        logger.info("Ingesting borrower_raw...")
        
        try:
            df.to_sql(
                'borrower_raw',
                self.engine,
                if_exists='replace',  # Use 'append' if data already exists
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"✅ Inserted {len(df)} records into borrower_raw")
        except Exception as e:
            logger.error(f"❌ Failed to insert borrower_raw: {e}")
            raise
    
    def ingest_loan_raw(self, df):
        """
        Insert loan training data
        """
        logger.info("Ingesting loan_raw...")
        
        try:
            df.to_sql(
                'loan_raw',
                self.engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"✅ Inserted {len(df)} records into loan_raw")
        except Exception as e:
            logger.error(f"❌ Failed to insert loan_raw: {e}")
            raise
    
    def ingest_borrower_prod(self, df):
        """
        Insert borrower production data
        """
        logger.info("Ingesting borrower_prod...")
        
        try:
            df.to_sql(
                'borrower_prod',
                self.engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"✅ Inserted {len(df)} records into borrower_prod")
        except Exception as e:
            logger.error(f"❌ Failed to insert borrower_prod: {e}")
            raise
    
    def ingest_loan_prod(self, df):
        """
        Insert loan production data
        """
        logger.info("Ingesting loan_prod...")
        
        try:
            df.to_sql(
                'loan_prod',
                self.engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            logger.info(f"✅ Inserted {len(df)} records into loan_prod")
        except Exception as e:
            logger.error(f"❌ Failed to insert loan_prod: {e}")
            raise
    
    def verify_ingestion(self):
        """
        Verify data ingestion
        """
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING DATA INGESTION")
        logger.info("=" * 60)
        
        tables = ['borrower_raw', 'loan_raw', 'borrower_prod', 'loan_prod']
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table};"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                count = result.fetchone()[0]
                logger.info(f"✅ {table}: {count} records")
        
        # Check loan status distribution
        query = "SELECT loan_status, COUNT(*) as count FROM loan_raw GROUP BY loan_status;"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            logger.info("\nLoan Status Distribution:")
            for row in result:
                logger.info(f"  {row[0]}: {row[1]}")
