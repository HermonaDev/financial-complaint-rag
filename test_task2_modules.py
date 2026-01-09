#!/usr/bin/env python3
"""Quick test of Task 2 modules."""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing Task 2 modules...")

# Test 1: Import modules
try:
    from sampling.stratified_sampler import StratifiedSampler
    from embedding.text_chunker import TextChunker
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create test data
print("\nCreating test data...")
test_data = {
    'Product': ['Credit card', 'Credit card', 'Savings account', 
                'Money transfers', 'Personal loan', 'Credit card'],
    'Complaint ID': ['001', '002', '003', '004', '005', '006'],
    'cleaned_narrative': [
        'This is a test complaint about credit card charges. ' * 10,
        'Short complaint.',
        'Another test about savings account issues. ' * 15,
        'Money transfer problem occurred. ' * 8,
        'Personal loan issue with payment. ' * 12,
        'Credit card fraud alert. ' * 5
    ]
}
test_df = pd.DataFrame(test_data)

print(f"Test DataFrame: {len(test_df)} rows")

# Test 3: Test sampler
print("\nTesting StratifiedSampler...")
sampler = StratifiedSampler(target_sample_size=4, random_state=42)
sample_sizes = sampler.calculate_stratified_size(test_df, 'Product')
print(f"Sample sizes: {sample_sizes}")

sampled_df = sampler.sample_complaints(test_df, 'Product')
print(f"Sampled DataFrame: {len(sampled_df)} rows")

# Test 4: Test chunker
print("\nTesting TextChunker...")
chunker = TextChunker(chunk_size=100, chunk_overlap=20)
chunks = chunker.chunk_dataframe(sampled_df, 'cleaned_narrative', 'Complaint ID', 'Product')
print(f"Generated {len(chunks)} chunks")

if chunks:
    chunks_df = chunker.chunks_to_dataframe(chunks)
    print(f"Chunks DataFrame: {len(chunks_df)} rows")
    print(f"Average chunk length: {chunks_df['text_length'].mean():.0f} chars")

print("\n✅ All tests passed!")
