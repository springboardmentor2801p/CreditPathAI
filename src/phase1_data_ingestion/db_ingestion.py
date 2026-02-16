# src/phase1_data_ingestion/db_ingestion.py

import pandas as pd
from sqlalchemy import text, inspect
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
    
    def clear_all_data(self):
        """
        Clear all data from tables while preserving structure
        """
        logger.info("Clearing existing data from all tables...")
        
        queries = [
            "DELETE FROM loan_prod;",
            "DELETE FROM loan_raw;",
            "DELETE FROM borrower_prod;",
            "DELETE FROM borrower_raw;"
        ]
        
        try:
            with self.engine.begin() as conn:
                for query in queries:
                    try:
                        conn.execute(text(query))
                        table_name = query.split()[2].rstrip(';')
                        logger.info(f"✅ Cleared {table_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clear table: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Clear operation warning: {e}")
    
    def table_exists(self, table_name):
        """
        Check if table exists
        """
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def ingest_dataframe(self, df, table_name, batch_size=5000):
        """
        Generic method to ingest dataframe into PostgreSQL with progress tracking
        """
        logger.info(f"Ingesting data into {table_name}...")
        
        try:
            total_rows = len(df)
            rows_inserted = 0
            
            # Insert in batches
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # Use multi method for faster insertion
                batch_df.to_sql(
                    table_name,
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi',
                    chunksize=1000
                )
                
                rows_inserted += len(batch_df)
                
                # Progress update every 10k rows or at completion
                if rows_inserted % 10000 == 0 or rows_inserted == total_rows:
                    logger.info(f"  Progress: {rows_inserted:,}/{total_rows:,} rows inserted ({rows_inserted*100//total_rows}%)")
            
            logger.info(f"✅ Successfully inserted {total_rows:,} records into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert into {table_name}")
            logger.error(f"   Error: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            
            # Additional debugging info
            logger.error(f"   DataFrame shape: {df.shape}")
            logger.error(f"   DataFrame columns: {list(df.columns)}")
            logger.error(f"   DataFrame dtypes:\n{df.dtypes}")
            
            raise
    
    def ingest_borrower_raw(self, df):
        """Insert borrower training data"""
        return self.ingest_dataframe(df, 'borrower_raw')
    
    def ingest_loan_raw(self, df):
        """Insert loan training data"""
        return self.ingest_dataframe(df, 'loan_raw')
    
    def ingest_borrower_prod(self, df):
        """Insert borrower production data"""
        return self.ingest_dataframe(df, 'borrower_prod', batch_size=100)
    
    def ingest_loan_prod(self, df):
        """Insert loan production data"""
        return self.ingest_dataframe(df, 'loan_prod', batch_size=100)
    
    def verify_ingestion(self):
        """
        Verify data ingestion with detailed statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("VERIFYING DATA INGESTION")
        logger.info("=" * 70)
        
        tables = ['borrower_raw', 'loan_raw', 'borrower_prod', 'loan_prod']
        
        try:
            with self.engine.connect() as conn:
                # Count records in each table
                for table in tables:
                    if not self.table_exists(table):
                        logger.warning(f"⚠️ Table {table} does not exist")
                        continue
                    
                    query = text(f"SELECT COUNT(*) as count FROM {table};")
                    result = conn.execute(query)
                    count = result.fetchone()[0]
                    logger.info(f"✅ {table:20s}: {count:,} records")
                
                # Loan status distribution
                if self.table_exists('loan_raw'):
                    logger.info("\n" + "-" * 70)
                    logger.info("Loan Status Distribution (Training Data):")
                    logger.info("-" * 70)
                    
                    query = text("""
                        SELECT 
                            loan_status, 
                            COUNT(*) as count,
                            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                        FROM loan_raw 
                        GROUP BY loan_status
                        ORDER BY count DESC;
                    """)
                    result = conn.execute(query)
                    
                    for row in result:
                        logger.info(f"  {row[0]:15s}: {row[1]:,} ({row[2]}%)")
                
                # Data integrity checks
                logger.info("\n" + "-" * 70)
                logger.info("Data Integrity Checks:")
                logger.info("-" * 70)
                
                # Check training data integrity
                query = text("""
                    SELECT COUNT(*) 
                    FROM loan_raw l 
                    LEFT JOIN borrower_raw b ON l.member_id = b.member_id 
                    WHERE b.member_id IS NULL;
                """)
                result = conn.execute(query)
                orphaned = result.fetchone()[0]
                
                if orphaned == 0:
                    logger.info(f"✅ All {self.get_count(conn, 'loan_raw'):,} loan records have matching borrowers")
                else:
                    logger.warning(f"⚠️ Found {orphaned:,} loan records without matching borrowers")
                
                # Check production data integrity
                if self.table_exists('loan_prod') and self.table_exists('borrower_prod'):
                    query = text("""
                        SELECT COUNT(*) 
                        FROM loan_prod l 
                        LEFT JOIN borrower_prod b ON l.member_id = b.member_id 
                        WHERE b.member_id IS NULL;
                    """)
                    result = conn.execute(query)
                    orphaned = result.fetchone()[0]
                    
                    if orphaned == 0:
                        logger.info(f"✅ All production loan records have matching borrowers")
                    else:
                        logger.warning(f"⚠️ Found {orphaned} production loans without matching borrowers")
                
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            raise
        
        logger.info("=" * 70)
    
    def get_count(self, conn, table_name):
        """Helper to get row count"""
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
        return result.fetchone()[0]
