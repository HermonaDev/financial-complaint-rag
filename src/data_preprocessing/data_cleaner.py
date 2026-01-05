"""Data cleaning utilities."""

import pandas as pd
import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and filters complaint data."""
    
    def __init__(self, target_products: List[str]):
        """Initialize with target product categories.
        
        Args:
            target_products: List of product categories to include
        """
        self.target_products = target_products
        
        # Common boilerplate patterns in complaint narratives
        self.boilerplate_patterns = [
            r"i am writing to file a complaint",
            r"dear sir/madam",
            r"to whom it may concern",
            r"please be advised that",
            r"this is to inform you that",
            r"i am writing regarding",
            r"i would like to report",
            r"i am writing this letter",
            r"i am writing to complain",
            r"i am writing to express",
        ]
    
    def filter_by_products(self, df: pd.DataFrame, product_col: str = "Product") -> pd.DataFrame:
        """Filter DataFrame to include only target products.
        
        Args:
            df: Input DataFrame
            product_col: Name of the product column
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)
        
        # Ensure product column exists
        if product_col not in df.columns:
            raise ValueError(f"Product column '{product_col}' not found in DataFrame")
        
        # Filter to target products
        df_filtered = df[df[product_col].isin(self.target_products)].copy()
        
        filtered_count = len(df_filtered)
        logger.info(f"Filtered by products: {initial_count} -> {filtered_count} records")
        logger.info(f"Products kept: {self.target_products}")
        
        return df_filtered
    
    def remove_empty_narratives(self, df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative") -> pd.DataFrame:
        """Remove records with empty or missing narratives.
        
        Args:
            df: Input DataFrame
            narrative_col: Name of the narrative column
            
        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)
        
        # Check if column exists
        if narrative_col not in df.columns:
            raise ValueError(f"Narrative column '{narrative_col}' not found in DataFrame")
        
        # Remove rows with empty narratives
        df_clean = df[df[narrative_col].notna() & (df[narrative_col].str.strip() != "")].copy()
        
        cleaned_count = len(df_clean)
        logger.info(f"Removed empty narratives: {initial_count} -> {cleaned_count} records")
        
        # Count narratives removed
        removed = initial_count - cleaned_count
        if removed > 0:
            logger.warning(f"Removed {removed} records with empty narratives")
        
        return df_clean
    
    def clean_narrative_text(self, text: str) -> str:
        """Clean individual narrative text.
        
        Args:
            text: Raw narrative text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove boilerplate phrases
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\-]', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_all_narratives(self, df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative") -> pd.DataFrame:
        """Clean all narrative texts in DataFrame.
        
        Args:
            df: Input DataFrame
            narrative_col: Name of the narrative column
            
        Returns:
            DataFrame with cleaned narratives
        """
        df_clean = df.copy()
        
        # Apply cleaning function
        df_clean[f"cleaned_{narrative_col}"] = df_clean[narrative_col].apply(self.clean_narrative_text)
        
        # Report on text length changes
        original_lengths = df[narrative_col].str.len().mean() if narrative_col in df.columns else 0
        cleaned_lengths = df_clean[f"cleaned_{narrative_col}"].str.len().mean()
        
        logger.info(f"Text cleaning completed")
        logger.info(f"Average original length: {original_lengths:.1f} characters")
        logger.info(f"Average cleaned length: {cleaned_lengths:.1f} characters")
        
        return df_clean
