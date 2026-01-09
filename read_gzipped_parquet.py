#!/usr/bin/env python3
"""Read gzipped parquet file."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import gzip

file_path = Path('data/raw/complaint_embeddings-001.parquet')
print(f"Reading gzipped parquet file: {file_path}")

# Method 1: Try with pyarrow specifying compression
try:
    print("\nMethod 1: Reading with pyarrow (auto-detect compression)...")
    table = pq.read_table(file_path)
    print(f"Success! Table shape: {table.num_rows} rows, {table.num_columns} columns")
    print(f"Column names: {table.column_names}")
    
    # Convert to pandas for inspection
    df = table.slice(0, 5).to_pandas()
    print("\nFirst 5 rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error with pyarrow: {e}")
    
    # Method 2: Try with pandas read_parquet
    try:
        print("\nMethod 2: Reading with pandas...")
        df = pd.read_parquet(file_path)
        print(f"Success! DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check embedding dimensions
        if 'embeddings' in df.columns:
            sample_embedding = df['embeddings'].iloc[0]
            print(f"\nSample embedding type: {type(sample_embedding)}")
            if hasattr(sample_embedding, 'shape'):
                print(f"Sample embedding shape: {sample_embedding.shape}")
            elif isinstance(sample_embedding, list):
                print(f"Sample embedding length: {len(sample_embedding)}")
                
    except Exception as e2:
        print(f"Error with pandas: {e2}")
        
        # Method 3: Manual decompression and reading
        try:
            print("\nMethod 3: Manual decompression...")
            with gzip.open(file_path, 'rb') as f:
                # Read first 1MB to see structure
                data = f.read(1024*1024)
                print(f"Decompressed {len(data)} bytes")
                
                # Try to read as parquet from memory
                import io
                buffer = io.BytesIO(data)
                # This might not work for partial file, but let's try
                try:
                    df = pd.read_parquet(buffer)
                    print(f"Success from buffer! Shape: {df.shape}")
                except:
                    print("Cannot read partial file as parquet")
                    # Save decompressed version
                    decompressed_path = file_path.with_suffix('.decompressed.parquet')
                    print(f"Saving decompressed version to {decompressed_path}")
                    with open(decompressed_path, 'wb') as out_f:
                        # Read and write in chunks
                        with gzip.open(file_path, 'rb') as gz_f:
                            chunk = gz_f.read(1024*1024)
                            while chunk:
                                out_f.write(chunk)
                                chunk = gz_f.read(1024*1024)
                    print(f"Decompressed file saved: {decompressed_path}")
                    
        except Exception as e3:
            print(f"Error with manual decompression: {e3}")

# Check file structure more carefully
print("\n" + "="*60)
print("File structure analysis:")

# Read just the schema
try:
    print("\nReading schema...")
    # Try with different compression options
    for comp in [None, 'gzip', 'snappy', 'brotli']:
        try:
            if comp:
                schema = pq.read_schema(file_path, compression=comp)
            else:
                schema = pq.read_schema(file_path)
            print(f"Schema with compression='{comp}': {schema}")
            break
        except:
            continue
except Exception as e:
    print(f"Cannot read schema: {e}")
