#!/usr/bin/env python3
"""Simplified Task 1 runner."""

import pandas as pd
from pathlib import Path
import json
import numpy as np

print("="*60)
print("Running Simplified Task 1")
print("="*60)

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Load and filter data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/complaints.csv', low_memory=False)
print(f"   Loaded {len(df):,} complaints")

# Filter to target products
print("\n2. Filtering to target products...")
target_products = ["Credit card", "Personal loan", "Savings account", "Money transfers"]
filtered = df[df['Product'].isin(target_products)].copy()
print(f"   After filtering: {len(filtered):,} complaints")

# Remove empty narratives
print("\n3. Removing empty narratives...")
initial_count = len(filtered)
filtered = filtered[filtered['Consumer complaint narrative'].notna() & 
                    (filtered['Consumer complaint narrative'].str.strip() != '')].copy()
print(f"   Removed {initial_count - len(filtered):,} empty narratives")
print(f"   Remaining: {len(filtered):,} complaints")

# Simple text cleaning
print("\n4. Cleaning text...")
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

filtered['cleaned_narrative'] = filtered['Consumer complaint narrative'].apply(clean_text)

# Basic EDA
print("\n5. Generating EDA summary...")
summary = {
    'total_records': int(len(filtered)),
    'product_distribution': {k: int(v) for k, v in filtered['Product'].value_counts().to_dict().items()},
    'narrative_stats': {
        'mean_length': float(filtered['cleaned_narrative'].str.len().mean()),
        'median_length': float(filtered['cleaned_narrative'].str.len().median()),
        'min_length': int(filtered['cleaned_narrative'].str.len().min()),
        'max_length': int(filtered['cleaned_narrative'].str.len().max()),
    }
}

# Save results
print("\n6. Saving results...")
output_dir = Path('data/processed')
output_dir.mkdir(exist_ok=True)

# Save filtered data
output_path = output_dir / 'filtered_complaints.csv'
filtered.to_csv(output_path, index=False)
print(f"   Saved filtered data to: {output_path}")
print(f"   File size: {output_path.stat().st_size / 1024**2:.1f} MB")

# Save summary
summary_path = output_dir / 'task1_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, cls=NumpyEncoder)
print(f"   Saved summary to: {summary_path}")

# Print summary
print("\n" + "="*60)
print("TASK 1 SUMMARY")
print("="*60)
print(f"Total complaints after processing: {summary['total_records']:,}")
print("\nProduct Distribution:")
for product, count in summary['product_distribution'].items():
    pct = (count / summary['total_records']) * 100
    print(f"  {product}: {count:,} ({pct:.1f}%)")
print(f"\nNarrative length:")
print(f"  Mean: {summary['narrative_stats']['mean_length']:.0f} chars")
print(f"  Median: {summary['narrative_stats']['median_length']:.0f} chars")
print(f"  Min: {summary['narrative_stats']['min_length']:,} chars")
print(f"  Max: {summary['narrative_stats']['max_length']:,} chars")
print("\n" + "="*60)
print("Task 1 completed successfully!")
print("="*60)