#!/usr/bin/env python3
"""Comprehensive Task 1 verification."""

import pandas as pd
from pathlib import Path
import json

print("="*60)
print("TASK 1 FINAL VERIFICATION")
print("="*60)

# Check 1: Filtered data exists
print("\n1. Checking filtered data...")
filtered_path = Path('data/processed/filtered_complaints.csv')
if filtered_path.exists():
    df = pd.read_csv(filtered_path)
    print(f"   ✅ Found: {len(df):,} rows")
    print(f"   File size: {filtered_path.stat().st_size / 1024**2:.1f} MB")
    
    # Check required columns
    required_cols = ['Product', 'Consumer complaint narrative']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"   ⚠ Missing columns: {missing}")
    else:
        print(f"   ✅ All required columns present")
        
    # Check product distribution
    print(f"\n   Product distribution:")
    product_counts = df['Product'].value_counts()
    for product, count in product_counts.items():
        pct = (count / len(df)) * 100
        print(f"     {product}: {count:,} ({pct:.1f}%)")
else:
    print("   ❌ Filtered data not found")

# Check 2: Summary exists and is valid
print("\n2. Checking summary file...")
summary_path = Path('data/processed/task1_summary.json')
if summary_path.exists():
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        print(f"   ✅ Valid JSON summary")
        print(f"   Total records: {summary['total_records']:,}")
        
        if summary['total_records'] == len(df):
            print("   ✅ Record count matches filtered data")
        else:
            print(f"   ⚠ Mismatch: summary={summary['total_records']:,}, data={len(df):,}")
            
    except Exception as e:
        print(f"   ❌ JSON error: {e}")
else:
    print("   ❌ Summary file not found")

# Check 3: Data quality
print("\n3. Checking data quality...")
if 'df' in locals():
    # Check for empty narratives
    if 'cleaned_narrative' in df.columns:
        empty_count = df['cleaned_narrative'].isna().sum() + (df['cleaned_narrative'] == '').sum()
        print(f"   Empty cleaned narratives: {empty_count}")
    else:
        print(f"   ⚠ 'cleaned_narrative' column not found")
        
    # Check original narratives
    narrative_col = 'Consumer complaint narrative'
    if narrative_col in df.columns:
        original_empty = df[narrative_col].isna().sum() + (df[narrative_col] == '').sum()
        print(f"   Empty original narratives: {original_empty}")
        print(f"   Non-empty narratives: {len(df) - original_empty:,}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
