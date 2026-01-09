#!/usr/bin/env python3
"""Verify embeddings file with proper pyarrow usage."""

import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

file_path = Path('data/raw/complaint_embeddings.parquet')
print(f"Verifying file: {file_path}")

if file_path.exists():
    print(f"✅ File exists")
    print(f"   Size: {file_path.stat().st_size / 1024**3:.2f} GB")
    
    try:
        # Read schema
        schema = pq.read_schema(file_path)
        print(f"\n📋 Schema:")
        for field in schema:
            print(f"   {field.name}: {field.type}")
        
        # Read first batch of rows
        print(f"\n📊 Reading first batch...")
        
        # Method 1: Use pyarrow to read first few rows
        table = pq.read_table(file_path)
        print(f"   Total rows in file: {table.num_rows:,}")
        print(f"   Total columns: {table.num_columns}")
        
        # Convert first 5 rows to pandas
        df = table.slice(0, 5).to_pandas()
        print(f"\n📝 First 5 rows:")
        print(df[['id', 'product_category', 'product']].head())
        
        # Check embedding dimension
        if 'embedding' in df.columns:
            sample_embedding = df['embedding'].iloc[0]
            print(f"\n🔢 Embedding info:")
            print(f"   Type: {type(sample_embedding)}")
            print(f"   Length: {len(sample_embedding)}")
            print(f"   Sample first 5 values: {sample_embedding[:5]}")
        
        # Check metadata structure
        if 'metadata' in df.columns:
            print(f"\n📋 Metadata structure:")
            sample_metadata = df['metadata'].iloc[0]
            print(f"   Type: {type(sample_metadata)}")
            if hasattr(sample_metadata, '__dict__'):
                print(f"   Keys: {list(sample_metadata.__dict__.keys())}")
            elif isinstance(sample_metadata, dict):
                print(f"   Keys: {list(sample_metadata.keys())}")
                for key, value in list(sample_metadata.items())[:5]:
                    print(f"     {key}: {value}")
        
        # Check product distribution
        print(f"\n📊 Checking product distribution...")
        
        # Read more rows for distribution
        table_larger = pq.read_table(file_path, columns=['product_category'])
        product_series = table_larger.column('product_category').to_pandas()
        
        print(f"   Total rows read for distribution: {len(product_series):,}")
        product_counts = product_series.value_counts()
        print(f"\n   Product Category Distribution:")
        for product, count in product_counts.items():
            percentage = (count / len(product_series)) * 100
            print(f"     {product}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n✅ File verification successful!")
        print(f"   Ready for Task 3 RAG implementation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"❌ File not found")
