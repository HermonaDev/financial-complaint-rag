#!/usr/bin/env python3
"""Proper verification of embeddings file with nested structure."""

import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import json

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
        
        # Read first few rows
        print(f"\n📊 Reading data...")
        table = pq.read_table(file_path)
        print(f"   Total rows: {table.num_rows:,}")
        print(f"   Total columns: {table.num_columns}")
        
        # Convert to pandas
        df = table.slice(0, 10).to_pandas()
        
        # The metadata is a struct, need to extract it
        print(f"\n📝 First 3 rows (showing structure):")
        for i in range(min(3, len(df))):
            print(f"\n   Row {i+1}:")
            print(f"     id: {df['id'].iloc[i]}")
            print(f"     document (first 100 chars): {df['document'].iloc[i][:100]}...")
            
            # Embedding info
            embedding = df['embedding'].iloc[i]
            print(f"     embedding: list of {len(embedding)} floats")
            print(f"       first 3 values: {embedding[:3]}")
            
            # Metadata - it's a pyarrow struct, convert to dict
            metadata = df['metadata'].iloc[i]
            print(f"     metadata:")
            if hasattr(metadata, 'as_py'):  # PyArrow StructScalar
                metadata_dict = metadata.as_py()
                for key, value in metadata_dict.items():
                    print(f"       {key}: {value}")
            elif isinstance(metadata, dict):
                for key, value in metadata.items():
                    print(f"       {key}: {value}")
        
        # Check product distribution by reading metadata column
        print(f"\n📊 Checking product distribution...")
        
        # Read just the metadata column for efficiency
        metadata_table = pq.read_table(
            file_path, 
            columns=['metadata']
        ).slice(0, 10000)  # Sample 10k rows for speed
        
        # Extract product_category from metadata
        product_categories = []
        for i in range(metadata_table.num_rows):
            metadata = metadata_table['metadata'][i]
            if hasattr(metadata, 'as_py'):
                metadata_dict = metadata.as_py()
                if 'product_category' in metadata_dict:
                    product_categories.append(metadata_dict['product_category'])
        
        if product_categories:
            print(f"   Sampled {len(product_categories)} rows")
            from collections import Counter
            counts = Counter(product_categories)
            print(f"\n   Product Category Distribution (sample):")
            for product, count in counts.most_common():
                percentage = (count / len(product_categories)) * 100
                print(f"     {product}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n✅ File structure confirmed!")
        print(f"   Ready for Task 3 implementation")
        
        # Save a small sample for testing
        sample_path = Path('data/processed/embeddings_sample.parquet')
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        table.slice(0, 1000).to_pandas().to_parquet(sample_path)
        print(f"\n💾 Saved 1000-row sample to: {sample_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
else:
    print(f"❌ File not found")
