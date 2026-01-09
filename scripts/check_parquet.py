#!/usr/bin/env python3
"""Check parquet file structure."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys

print("Checking parquet files...")

# Check complaint_embeddings-001.parquet
emb_path = Path('data/raw/complaint_embeddings-001.parquet')
if emb_path.exists():
    print(f"File exists: {emb_path}")
    print(f"File size: {emb_path.stat().st_size / 1024**2:.1f} MB")
    
    # Try to read metadata first
    try:
        # Read just the schema
        schema = pq.read_schema(emb_path)
        print(f"Schema: {schema}")
        
        # Try to read first few rows with pyarrow
        table = pq.read_table(emb_path)
        print(f"Table shape: {table.num_rows} rows, {table.num_columns} columns")
        print(f"Column names: {table.column_names}")
        
        # Convert to pandas (just first 5 rows)
        df = table.slice(0, 5).to_pandas()
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Check column types
        print(f"\nColumn dtypes:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
            
    except Exception as e:
        print(f"Error reading parquet: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"File not found: {emb_path}")

# Check for other parquet files
print("\nLooking for other parquet files...")
for f in Path('data/raw').glob('*.parquet'):
    if f != emb_path:
        print(f"Found: {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")
