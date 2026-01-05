"""Data loading utilities."""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import yaml
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of complaint datasets."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.config['data']['raw_path'])
        self.processed_data_path = Path(self.config['data']['processed_path'])
        
        # Ensure directories exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
    def load_raw_complaints(self, file_name: str = "complaints.csv") -> pd.DataFrame:
        """Load raw complaint data from CSV.
        
        Args:
            file_name: Name of the CSV file in raw data directory
            
        Returns:
            DataFrame containing raw complaint data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = self.raw_data_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        
        logger.info(f"Loading raw data from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df)} raw complaints")
        
        return df
    
    def load_embeddings(self, file_name: str = "complaint_embeddings.parquet") -> pd.DataFrame:
        """Load pre-built embeddings.
        
        Args:
            file_name: Name of the parquet file in raw data directory
            
        Returns:
            DataFrame containing embeddings and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = self.raw_data_path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
        logger.info(f"Loading embeddings from {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} embedding records")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, file_name: str = "filtered_complaints.csv") -> Path:
        """Save processed DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            file_name: Name of the output file
            
        Returns:
            Path to saved file
        """
        file_path = self.processed_data_path / file_name
        df.to_csv(file_path, index=False)
        logger.info(f"Saved processed data to {file_path}")
        return file_path
