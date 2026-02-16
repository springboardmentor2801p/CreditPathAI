# database/config/db_config.py

import os
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import psycopg2
from urllib.parse import quote_plus

class DatabaseConfig:
    """
    Database configuration and connection management
    """
    
    def __init__(self):
        # Database credentials (use environment variables in production)
        self.DB_HOST = os.getenv('DB_HOST', 'localhost')
        self.DB_PORT = os.getenv('DB_PORT', '5432')
        self.DB_NAME = os.getenv('DB_NAME', 'creditpathai')
        self.DB_USER = os.getenv('DB_USER', 'postgres')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD', '7432')
        
    def get_connection_string(self):
        """
        Generate PostgreSQL connection string for SQLAlchemy
        """
        password = quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    def create_engine(self):
        """
        Create SQLAlchemy engine with connection pooling
        """
        connection_string = self.get_connection_string()
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL debugging
        )
        return engine
    
    def get_psycopg2_connection(self):
        """
        Create direct psycopg2 connection (for specific use cases)
        """
        conn = psycopg2.connect(
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD
        )
        return conn
    
    def test_connection(self):
        """
        Test database connection
        """
        try:
            engine = self.create_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print(f"✅ Database connection successful!")
                print(f"PostgreSQL version: {version}")
                return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

# Singleton instance
db_config = DatabaseConfig()
