# src/phase1_data_ingestion/reset_database.py

import sys
sys.path.append('.')

from database.config.db_config import db_config
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database():
    """
    Drop and recreate all tables (complete reset)
    """
    logger.info("Resetting database...")
    
    engine = db_config.create_engine()
    
    drop_queries = [
        "DROP TABLE IF EXISTS risk_scores CASCADE;",
        "DROP TABLE IF EXISTS features_training CASCADE;",
        "DROP TABLE IF EXISTS loan_prod CASCADE;",
        "DROP TABLE IF EXISTS loan_raw CASCADE;",
        "DROP TABLE IF EXISTS borrower_prod CASCADE;",
        "DROP TABLE IF EXISTS borrower_raw CASCADE;"
    ]
    
    try:
        with engine.connect() as conn:
            for query in drop_queries:
                conn.execute(text(query))
                conn.commit()
            logger.info("✅ All tables dropped successfully")
        
        logger.info("\nPlease run the DDL schema script again:")
        logger.info("psql -U postgres -d creditpathai -f database\\schema\\ddl_schema.sql")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    reset_database()
