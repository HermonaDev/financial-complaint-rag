#!/usr/bin/env python3
"""Filter CFPB data to our 4 target product categories - FINAL CORRECTED VERSION."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def map_product_to_category(product_name: str) -> str:
    """Map CFPB product names to our 4 standardized categories."""
    if not isinstance(product_name, str):
        return None
    
    product_lower = product_name.lower()
    
    # Credit card category
    if any(keyword in product_lower for keyword in ['credit card', 'credit card or prepaid card', 'credit card or prepaid']):
        return 'Credit card'
    
    # Personal loan category
    if any(keyword in product_lower for keyword in ['personal loan', 'payday loan', 'title loan', 'consumer loan', 'payday loan, title loan']):
        return 'Personal loan'
    
    # Savings account category
    if any(keyword in product_lower for keyword in ['savings account', 'checking account', 'bank account', 'checking or savings', 'bank account or service']):
        return 'Savings account'
    
    # Money transfers category
    if any(keyword in product_lower for keyword in ['money transfer', 'money transfers', 'remittance', 'money transfer, virtual currency']):
        return 'Money transfers'
    
    return None


def filter_and_clean_complaints(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Filter complaints to target products and clean the data."""
    logger.info(f"Loading data from {input_path}")
    
    # Load data in chunks
    chunks = []
    for chunk in pd.read_csv(input_path, low_memory=False, chunksize=100000):
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df):,} raw complaints")
    
    # Map products to our categories
    logger.info("Mapping products to standardized categories...")
    df['product_category'] = df['Product'].apply(map_product_to_category)
    
    # Filter to only our target categories
    initial_count = len(df)
    df = df[df['product_category'].notna()].copy()
    logger.info(f"Filtered to target categories: {initial_count:,} → {len(df):,}")
    
    # Show distribution
    category_counts = df['product_category'].value_counts()
    logger.info("Category distribution after filtering:")
    for category, count in category_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {category}: {count:,} ({pct:.1f}%)")
    
    # Remove empty narratives
    initial_count = len(df)
    df = df[df['Consumer complaint narrative'].notna() & 
            (df['Consumer complaint narrative'].str.strip() != '')].copy()
    logger.info(f"Removed empty narratives: {initial_count:,} → {len(df):,}")
    
    # Clean text narratives
    logger.info("Cleaning narrative texts...")
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # Standardize column names
    df['product_original'] = df['Product']
    df['Product'] = df['product_category']
    df = df.drop(columns=['product_category'])
    
    # Select columns
    keep_columns = [
        'Product', 'product_original', 'Consumer complaint narrative', 'cleaned_narrative',
        'Issue', 'Sub-issue', 'Company', 'State', 'Date received',
        'Complaint ID', 'Company response to consumer'
    ]
    available_columns = [col for col in keep_columns if col in df.columns]
    df = df[available_columns]
    
    # Save
    logger.info(f"Saving filtered data to {output_path}")
    df.to_csv(output_path, index=False)
    
    return df


def create_summary(df: pd.DataFrame, summary_path: Path, raw_count: int):
    """Create summary of filtered data."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    summary = {
        'total_records': int(len(df)),
        'raw_records': int(raw_count),
        'product_distribution': {str(k): int(v) for k, v in df['Product'].value_counts().to_dict().items()},
        'narrative_stats': {
            'mean_length': float(df['cleaned_narrative'].str.len().mean()),
            'median_length': float(df['cleaned_narrative'].str.len().median()),
            'min_length': int(df['cleaned_narrative'].str.len().min()),
            'max_length': int(df['cleaned_narrative'].str.len().max()),
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Summary saved to {summary_path}")
    return summary


def main():
    """Main execution function."""
    input_path = Path('data/raw/complaints.csv')
    output_path = Path('data/processed/filtered_complaints.csv')
    summary_path = Path('data/processed/task1_summary.json')
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    output_path.parent.mkdir(exist_ok=True)
    
    # Get raw count first
    raw_count = 0
    for chunk in pd.read_csv(input_path, low_memory=False, chunksize=100000):
        raw_count += len(chunk)
    
    # Filter and clean
    df = filter_and_clean_complaints(input_path, output_path)
    
    # Create summary
    summary = create_summary(df, summary_path, raw_count)
    
    # Print final summary
    print("\n" + "="*60)
    print("TASK 1 COMPLETE - FINAL DATA READY")
    print("="*60)
    print(f"Raw complaints: {raw_count:,}")
    print(f"Filtered complaints: {len(df):,}")
    print(f"Reduction: {(1 - len(df)/raw_count)*100:.1f}%")
    
    print("\nProduct Distribution:")
    for product, count in summary['product_distribution'].items():
        pct = (count / len(df)) * 100
        print(f"  {product}: {count:,} ({pct:.1f}%)")
    
    print("\nNarrative Statistics:")
    stats = summary['narrative_stats']
    print(f"  Mean length: {stats['mean_length']:.0f} chars")
    print(f"  Median length: {stats['median_length']:.0f} chars")
    print(f"  Min length: {stats['min_length']:,} chars")
    print(f"  Max length: {stats['max_length']:,} chars")
    
    print("\nFile Information:")
    print(f"  Filtered data: {output_path.stat().st_size / 1024**2:.1f} MB")
    print("\n" + "="*60)
    print("✅ READY FOR TASK 2: Text Chunking and Embedding")
    print("="*60)


if __name__ == "__main__":
    main()