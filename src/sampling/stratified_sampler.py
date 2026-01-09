"""Stratified sampling for complaint data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StratifiedSampler:
    """Creates stratified samples from complaint data."""
    
    def __init__(self, target_sample_size: int = 12000, random_state: int = 42):
        """Initialize sampler with target sample size.
        
        Args:
            target_sample_size: Total number of complaints to sample
            random_state: Random seed for reproducibility
        """
        self.target_sample_size = target_sample_size
        self.random_state = random_state
        np.random.seed(random_state)
    
    def calculate_stratified_size(self, df: pd.DataFrame, product_col: str = 'Product') -> Dict[str, int]:
        """Calculate sample size per product category.
        
        Args:
            df: Input DataFrame with complaint data
            product_col: Column containing product categories
            
        Returns:
            Dictionary mapping product categories to sample sizes
        """
        # Get product distribution
        product_counts = df[product_col].value_counts()
        total_complaints = len(df)
        
        # Calculate proportional allocation
        sample_sizes = {}
        for product, count in product_counts.items():
            proportion = count / total_complaints
            sample_sizes[product] = int(proportion * self.target_sample_size)
        
        # Adjust for rounding errors to ensure total equals target
        total_allocated = sum(sample_sizes.values())
        difference = self.target_sample_size - total_allocated
        
        if difference != 0:
            # Add difference to the largest category
            largest_product = max(sample_sizes.items(), key=lambda x: x[1])[0]
            sample_sizes[largest_product] += difference
        
        logger.info(f"Stratified sample sizes:")
        for product, size in sample_sizes.items():
            logger.info(f"  {product}: {size} samples")
        
        return sample_sizes
    
    def sample_complaints(self, df: pd.DataFrame, product_col: str = 'Product') -> pd.DataFrame:
        """Create stratified sample from complaint data.
        
        Args:
            df: Input DataFrame with complaint data
            product_col: Column containing product categories
            
        Returns:
            Stratified sample DataFrame
        """
        logger.info(f"Creating stratified sample of {self.target_sample_size:,} complaints...")
        
        # Calculate sample sizes per product
        sample_sizes = self.calculate_stratified_size(df, product_col)
        
        # Sample from each product category
        sampled_dfs = []
        for product, size in sample_sizes.items():
            product_df = df[df[product_col] == product]
            
            if len(product_df) < size:
                logger.warning(f"Product '{product}' has only {len(product_df)} complaints, "
                              f"sampling all available")
                size = len(product_df)
            
            if size > 0:
                product_sample = product_df.sample(n=size, random_state=self.random_state)
                sampled_dfs.append(product_sample)
                logger.info(f"  Sampled {size:,} from {product} "
                           f"(had {len(product_df):,} available)")
        
        # Combine samples
        if sampled_dfs:
            sampled_df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"Created stratified sample with {len(sampled_df):,} total complaints")
            
            # Verify distribution
            final_distribution = sampled_df[product_col].value_counts()
            logger.info("Final sample distribution:")
            for product, count in final_distribution.items():
                pct = (count / len(sampled_df)) * 100
                logger.info(f"  {product}: {count:,} ({pct:.1f}%)")
            
            return sampled_df
        else:
            logger.error("No samples created")
            return pd.DataFrame()
    
    def save_sample(self, sampled_df: pd.DataFrame, output_path: Path):
        """Save sampled data to CSV.
        
        Args:
            sampled_df: Sampled DataFrame
            output_path: Path to save the sample
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(output_path, index=False)
        logger.info(f"Saved sample to {output_path}")
        logger.info(f"Sample size: {len(sampled_df):,} rows")
        logger.info(f"File size: {output_path.stat().st_size / 1024**2:.1f} MB")


def load_complaints_data(input_path: Path) -> pd.DataFrame:
    """Load complaint data from CSV.
    
    Args:
        input_path: Path to complaint data CSV
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading complaint data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} complaints")
    return df


def create_sample_report(sampled_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict:
    """Create report comparing sample to original data.
    
    Args:
        sampled_df: Sampled DataFrame
        original_df: Original DataFrame
        
    Returns:
        Dictionary with comparison statistics
    """
    report = {
        'sample_size': len(sampled_df),
        'original_size': len(original_df),
        'sampling_rate': len(sampled_df) / len(original_df),
        'product_distribution': {},
        'narrative_stats': {}
    }
    
    # Compare product distributions
    original_dist = original_df['Product'].value_counts(normalize=True)
    sample_dist = sampled_df['Product'].value_counts(normalize=True)
    
    for product in set(original_dist.index) | set(sample_dist.index):
        report['product_distribution'][product] = {
            'original_pct': original_dist.get(product, 0) * 100,
            'sample_pct': sample_dist.get(product, 0) * 100,
            'difference': (sample_dist.get(product, 0) - original_dist.get(product, 0)) * 100
        }
    
    # Compare narrative lengths
    if 'cleaned_narrative' in sampled_df.columns:
        report['narrative_stats']['sample'] = {
            'mean_length': sampled_df['cleaned_narrative'].str.len().mean(),
            'median_length': sampled_df['cleaned_narrative'].str.len().median()
        }
    
    if 'cleaned_narrative' in original_df.columns:
        report['narrative_stats']['original'] = {
            'mean_length': original_df['cleaned_narrative'].str.len().mean(),
            'median_length': original_df['cleaned_narrative'].str.len().median()
        }
    
    return report