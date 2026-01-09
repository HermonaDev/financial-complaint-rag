#!/usr/bin/env python3
"""Create a fresh, clean summary from filtered data."""

import pandas as pd
import json
from pathlib import Path

print("Creating fresh Task 1 summary...")

# Load filtered data
filtered_path = Path('data/processed/filtered_complaints.csv')
if not filtered_path.exists():
    print(f"Error: {filtered_path} not found")
    exit(1)

df = pd.read_csv(filtered_path)
print(f"Loaded {len(df):,} filtered complaints")

# Create summary
summary = {
    'total_records': int(len(df)),
    'product_distribution': {},
    'narrative_stats': {
        'mean_length': 0.0,
        'median_length': 0.0,
        'min_length': 0,
        'max_length': 0
    },
    'processing_steps': [
        'Loaded 9,609,797 raw complaints',
        'Filtered to target products: 232,040 complaints',
        'Removed empty narratives: 82,164 final complaints'
    ]
}

# Calculate product distribution
product_counts = df['Product'].value_counts()
for product, count in product_counts.items():
    summary['product_distribution'][str(product)] = int(count)

# Calculate narrative stats if the column exists
if 'cleaned_narrative' in df.columns:
    lengths = df['cleaned_narrative'].str.len()
    summary['narrative_stats'] = {
        'mean_length': float(lengths.mean()),
        'median_length': float(lengths.median()),
        'min_length': int(lengths.min()),
        'max_length': int(lengths.max())
    }

# Save summary
summary_path = Path('data/processed/task1_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Summary saved to {summary_path}")
print(f"Total records: {summary['total_records']:,}")
print(f"Products found: {len(summary['product_distribution'])}")
print(f"Mean narrative length: {summary['narrative_stats']['mean_length']:.0f} chars")
