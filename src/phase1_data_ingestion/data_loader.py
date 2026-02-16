# src/phase1_data_ingestion/data_loader.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Load and parse raw data files
    """
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        
    def load_borrower(self, filename='Borrower.txt'):
        """
        Load borrower training data
        """
        logger.info(f"Loading {filename}...")
        filepath = self.data_dir / filename
        
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8',
            low_memory=False
        )
        
        logger.info(f"✅ Loaded {len(df)} records from {filename}")
        logger.info(f"Columns: {list(df.columns)}")
        return df
    
    def load_loan(self, filename='Loan.txt'):
        """
        Load loan training data
        """
        logger.info(f"Loading {filename}...")
        filepath = self.data_dir / filename
        
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8',
            low_memory=False
        )
        
        logger.info(f"✅ Loaded {len(df)} records from {filename}")
        logger.info(f"Loan Status distribution:\n{df['loanStatus'].value_counts()}")
        return df
    
    def load_borrower_prod(self, filename='Borrower_Prod.txt'):
        """
        Load borrower production data
        """
        logger.info(f"Loading {filename}...")
        filepath = self.data_dir / filename
        
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8'
        )
        
        logger.info(f"✅ Loaded {len(df)} records from {filename}")
        return df
    
    def load_loan_prod(self, filename='Loan_Prod.txt'):
        """
        Load loan production data
        """
        logger.info(f"Loading {filename}...")
        filepath = self.data_dir / filename
        
        df = pd.read_csv(
            filepath,
            sep='\t',
            encoding='utf-8'
        )
        
        logger.info(f"✅ Loaded {len(df)} records from {filename}")
        return df
    
    def load_all(self):
        """
        Load all datasets
        """
        datasets = {
            'borrower': self.load_borrower(),
            'loan': self.load_loan(),
            'borrower_prod': self.load_borrower_prod(),
            'loan_prod': self.load_loan_prod()
        }
        
        logger.info("=" * 60)
        logger.info("✅ All datasets loaded successfully!")
        logger.info("=" * 60)
        
        return datasets
