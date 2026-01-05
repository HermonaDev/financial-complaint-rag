#!/usr/bin/env python3
"""
Task 1: EDA and Data Preprocessing
Main script for exploratory data analysis and data cleaning.
"""

import sys
from pathlib import Path

# Add src to path before importing local package
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import json
import pandas as pd

from data_preprocessing import DataLoader, DataCleaner, DataAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task_1_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def generate_eda_report(df: pd.DataFrame, summary: dict, output_path: Path):
    """Generate a textual EDA report.
    
    Args:
        df: Processed DataFrame
        summary: EDA summary dictionary
        output_path: Path to saved processed data
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("EDA AND DATA PREPROCESSING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset Overview
    report_lines.append("1. DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total records after processing: {len(df):,}")
    report_lines.append(f"Total columns: {len(df.columns)}")
    report_lines.append("")
    
    # Product Distribution
    report_lines.append("2. PRODUCT DISTRIBUTION")
    report_lines.append("-" * 40)
    product_counts = summary.get('product_distribution', {}).get('counts', {})
    for product, count in product_counts.items():
        percentage = (count / len(df)) * 100
        report_lines.append(f"{product}: {count:,} complaints ({percentage:.1f}%)")
    report_lines.append("")
    
    # Narrative Analysis
    report_lines.append("3. NARRATIVE ANALYSIS")
    report_lines.append("-" * 40)
    nar_stats = summary.get('narrative_analysis', {})
    report_lines.append(f"Mean narrative length: {nar_stats.get('mean', 0):.0f} characters")
    report_lines.append(f"Median narrative length: {nar_stats.get('median', 0):.0f} characters")
    report_lines.append(
        f"Shortest narrative: {nar_stats.get('min', 0):,} characters"
    )
    report_lines.append(
        f"Longest narrative: {nar_stats.get('max', 0):,} characters"
    )
    short_thresh = nar_stats.get('short_threshold', 0)
    short_count = nar_stats.get('very_short_count', 0)
    long_thresh = nar_stats.get('long_threshold', 0)
    long_count = nar_stats.get('very_long_count', 0)
    report_lines.append(
        f"Very short narratives (< {short_thresh:.0f} chars): {short_count:,}"
    )
    report_lines.append(
        f"Very long narratives (> {long_thresh:.0f} chars): {long_count:,}"
    )
    report_lines.append("")
    
    # Data Quality
    report_lines.append("4. DATA QUALITY")
    report_lines.append("-" * 40)
    missing = summary.get('missing_values', {})
    for col, count in missing.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            report_lines.append(f"{col}: {count:,} missing values ({percentage:.1f}%)")
    
    if all(v == 0 for v in missing.values()):
        report_lines.append("No missing values in key columns")
    report_lines.append("")
    
    # Output Information
    report_lines.append("5. OUTPUT FILES")
    report_lines.append("-" * 40)
    report_lines.append(f"Processed data: {output_path}")
    report_lines.append("EDA plots saved to: notebooks/eda_output/")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_path = output_path.parent / "eda_report.txt"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"EDA report saved to {report_path}")
    
    # Also print to console
    print("\n".join(report_lines))


def main():
    """Main execution function for Task 1."""
    logger.info("Starting Task 1: EDA and Data Preprocessing")
    
    try:
        # Initialize components
        logger.info("Initializing data processing components...")
        loader = DataLoader()
        cleaner = DataCleaner(target_products=[
            "Credit card", 
            "Personal loan", 
            "Savings account", 
            "Money transfers"
        ])
        analyzer = DataAnalyzer()
        
        # Step 1: Load raw data
        logger.info("Step 1: Loading raw complaint data...")
        raw_df = loader.load_raw_complaints()
        logger.info(f"Raw data shape: {raw_df.shape}")
        
        # Step 2: Initial EDA on raw data
        logger.info("Step 2: Performing initial EDA on raw data...")
        raw_summary = analyzer.generate_summary_report(
            raw_df, 
            product_col="Product",
            narrative_col="Consumer complaint narrative"
        )
        
        # Save raw EDA summary
        with open(loader.processed_data_path / "raw_eda_summary.json", 'w') as f:
            json.dump(raw_summary, f, indent=2, default=str)
        
        logger.info(f"Raw EDA summary saved to {loader.processed_data_path / 'raw_eda_summary.json'}")
        
        # Step 3: Filter data by products
        logger.info("Step 3: Filtering data by target products...")
        filtered_df = cleaner.filter_by_products(raw_df, product_col="Product")
        
        # Step 4: Remove empty narratives
        logger.info("Step 4: Removing empty narratives...")
        filtered_df = cleaner.remove_empty_narratives(
            filtered_df, 
            narrative_col="Consumer complaint narrative"
        )
        
        # Step 5: Clean narrative texts
        logger.info("Step 5: Cleaning narrative texts...")
        cleaned_df = cleaner.clean_all_narratives(
            filtered_df, 
            narrative_col="Consumer complaint narrative"
        )
        
        # Step 6: Final EDA on cleaned data
        logger.info("Step 6: Performing EDA on cleaned data...")
        final_summary = analyzer.generate_summary_report(
            cleaned_df, 
            product_col="Product",
            narrative_col="cleaned_Consumer complaint narrative"
        )
        
        # Save final EDA summary
        with open(loader.processed_data_path / "final_eda_summary.json", 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        logger.info(f"Final EDA summary saved to {loader.processed_data_path / 'final_eda_summary.json'}")
        
        # Step 7: Save processed data
        logger.info("Step 7: Saving processed data...")
        output_path = loader.save_processed_data(cleaned_df, "filtered_complaints.csv")
        
        # Step 8: Generate EDA report text
        logger.info("Step 8: Generating EDA report...")
        generate_eda_report(cleaned_df, final_summary, output_path)
        
        logger.info("Task 1 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Task 1: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
