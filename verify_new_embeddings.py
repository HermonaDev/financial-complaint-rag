#!/usr/bin/env python3
"""Verify the new embeddings file."""

import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

file_path = Path('data/raw/complaint_embeddings.parquet')
print(f"Verifying file: {file_path}")

if file_path.exists():
    print(f"✅ File exists")
    print(f"   Size: {file_path.stat().st_size / 1024**3:.2f} GB")
    
    # Try to read metadata first
    try:
        # Read schema
        schema = pq.read_schema(file_path)
        print(f"\n📋 Schema:")
        for field in schema:
            print(f"   {field.name}: {field.type}")
        
        # Read first few rows
        print(f"\n📊 Reading first 5 rows...")
        df = pd.read_parquet(file_path, nrows=5)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📝 Sample data:")
        for i, row in df.iterrows():
            print(f"\n   Row {i+1}:")
            for col in df.columns:
                value = row[col]
                if col == 'embeddings' and isinstance(value, (list, np.ndarray)):
                    print(f"     {col}: {type(value).__name__} with length {len(value)}")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"     {col}: {value[:100]}...")
                else:
                    print(f"     {col}: {value}")
        
        # Check unique values in key columns
        print(f"\n🔍 Unique values in key columns:")
        for col in ['product_category', 'product', 'issue']:
            if col in df.columns:
                unique_vals = df[col].unique()[:5]
                print(f"   {col}: {len(df[col].unique())} unique values, sample: {list(unique_vals)}")
                
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"❌ File not found: {file_path}")
    
    # List available files
    print("\n📁 Files in data/raw/:")
    for f in Path('data/raw').iterdir():
        if 'embedding' in str(f).lower() or 'parquet' in str(f).lower():
            print(f"   - {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")
