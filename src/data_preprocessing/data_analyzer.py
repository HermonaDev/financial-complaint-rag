"""EDA and data analysis utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Performs exploratory data analysis on complaint data."""
    
    def __init__(self, output_dir: Path = Path("notebooks/eda_output")):
        """Initialize with output directory for plots.
        
        Args:
            output_dir: Directory to save EDA visualizations
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_product_distribution(self, df: pd.DataFrame, product_col: str = "Product") -> Dict:
        """Analyze distribution of complaints across products.
        
        Args:
            df: Input DataFrame
            product_col: Name of the product column
            
        Returns:
            Dictionary with distribution statistics
        """
        if product_col not in df.columns:
            raise ValueError(f"Product column '{product_col}' not found in DataFrame")
        
        # Calculate distribution
        product_dist = df[product_col].value_counts()
        product_percent = df[product_col].value_counts(normalize=True) * 100
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        product_dist.plot(kind='bar')
        plt.title('Complaint Distribution by Product')
        plt.xlabel('Product')
        plt.ylabel('Number of Complaints')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "product_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Product distribution plot saved to {plot_path}")
        
        return {
            'counts': product_dist.to_dict(),
            'percentages': product_percent.round(2).to_dict(),
            'plot_path': plot_path
        }
    
    def analyze_narrative_lengths(self, df: pd.DataFrame, narrative_col: str = "Consumer complaint narrative") -> Dict:
        """Analyze length distribution of complaint narratives.
        
        Args:
            df: Input DataFrame
            narrative_col: Name of the narrative column
            
        Returns:
            Dictionary with length statistics
        """
        if narrative_col not in df.columns:
            raise ValueError(f"Narrative column '{narrative_col}' not found in DataFrame")
        
        # Calculate narrative lengths
        df = df.copy()
        df['narrative_length'] = df[narrative_col].astype(str).apply(len)
        
        # Calculate statistics
        length_stats = {
            'mean': df['narrative_length'].mean(),
            'median': df['narrative_length'].median(),
            'std': df['narrative_length'].std(),
            'min': df['narrative_length'].min(),
            'max': df['narrative_length'].max(),
            'total_empty': df[narrative_col].isna().sum() + (df[narrative_col] == '').sum()
        }
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(df['narrative_length'], bins=50, edgecolor='black')
        axes[0].axvline(length_stats['mean'], color='red', linestyle='--', label=f"Mean: {length_stats['mean']:.0f}")
        axes[0].axvline(length_stats['median'], color='green', linestyle='--', label=f"Median: {length_stats['median']:.0f}")
        axes[0].set_title('Distribution of Narrative Lengths')
        axes[0].set_xlabel('Number of Characters')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(df['narrative_length'])
        axes[1].set_title('Box Plot of Narrative Lengths')
        axes[1].set_ylabel('Number of Characters')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "narrative_lengths.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Narrative length analysis plot saved to {plot_path}")
        
        # Identify outliers (very short or very long)
        Q1 = df['narrative_length'].quantile(0.25)
        Q3 = df['narrative_length'].quantile(0.75)
        IQR = Q3 - Q1
        
        short_threshold = Q1 - 1.5 * IQR
        long_threshold = Q3 + 1.5 * IQR
        
        very_short = df[df['narrative_length'] < max(0, short_threshold)]
        very_long = df[df['narrative_length'] > long_threshold]
        
        length_stats['very_short_count'] = len(very_short)
        length_stats['very_long_count'] = len(very_long)
        length_stats['short_threshold'] = short_threshold
        length_stats['long_threshold'] = long_threshold
        
        return length_stats
    
    def generate_summary_report(self, df: pd.DataFrame, 
                                product_col: str = "Product",
                                narrative_col: str = "Consumer complaint narrative") -> Dict:
        """Generate comprehensive EDA summary report.
        
        Args:
            df: Input DataFrame
            product_col: Name of the product column
            narrative_col: Name of the narrative column
            
        Returns:
            Dictionary with complete EDA summary
        """
        report = {}
        
        # Basic statistics
        report['total_records'] = len(df)
        report['columns'] = list(df.columns)
        report['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Missing values
        report['missing_values'] = df.isnull().sum().to_dict()
        report['missing_percentage'] = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        
        # Product distribution
        report['product_distribution'] = self.analyze_product_distribution(df, product_col)
        
        # Narrative analysis
        report['narrative_analysis'] = self.analyze_narrative_lengths(df, narrative_col)
        
        # Date analysis (if date column exists)
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            report['date_range'] = {
                'min': df[date_col].min(),
                'max': df[date_col].max(),
                'missing_dates': df[date_col].isna().sum()
            }
        
        logger.info("EDA summary report generated")
        
        return report
