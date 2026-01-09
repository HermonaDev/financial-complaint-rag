#!/usr/bin/env python3
"""Quick script to verify data exists and can be loaded."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from data_preprocessing import DataLoader

    print("Testing data loading...")

    loader = DataLoader()

    # Try to load complaints
    print("1. Loading complaints.csv...")
    complaints = loader.load_raw_complaints()
    print("   ✓ Loaded {} complaints".format(len(complaints)))
    print("   Columns: {}".format(list(complaints.columns)))
    
    # Try to load embeddings if they exist
    if (Path(loader.raw_data_path) / "complaint_embeddings.parquet").exists():
        print("\n2. Loading embeddings...")
        embeddings = loader.load_embeddings()
        print(f"   ✓ Loaded {len(embeddings)} embedding records")
    else:
        print("\n2. Embeddings file not found (this is expected for Task 1)")
    
    print("\n✓ Data loading verified successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
