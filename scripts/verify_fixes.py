#!/usr/bin/env python3
"""Verify all fixes are working."""

import sys
from pathlib import Path

print("=" * 60)
print("Verifying Fixes")
print("=" * 60)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    print("\n1. Testing imports...")
    from data_preprocessing import DataLoader, DataCleaner, DataAnalyzer
    print("✓ All imports successful")
    
    print("\n2. Testing DataLoader...")
    loader = DataLoader()
    print("✓ DataLoader initialized")
    
    # Check config
    assert hasattr(loader, 'raw_data_path')
    assert hasattr(loader, 'processed_data_path')
    print("  Config loaded correctly")
    
    print("\n3. Testing DataCleaner...")
    target_products = [
        "Credit card",
        "Personal loan",
        "Savings account",
        "Money transfers",
    ]
    cleaner = DataCleaner(target_products)
    print("✓ DataCleaner initialized with {} target products".format(len(target_products)))
    
    print("\n4. Testing DataAnalyzer...")
    analyzer = DataAnalyzer()
    print("✓ DataAnalyzer initialized")
    
    print("\n5. Testing with sample data...")
    import pandas as pd
    import numpy as np
    
    # Create test data
    test_data = {
        'Product': ['Credit card', 'Personal loan', 'Credit card', 'Other'],
        'Consumer complaint narrative': [
            'Test complaint about credit card.',
            'Loan issue here.',
            '',
            np.nan
        ]
    }
    test_df = pd.DataFrame(test_data)
    
    # Test filtering
    filtered = cleaner.filter_by_products(test_df, 'Product')
    print("  Filter test: {} -> {} rows".format(len(test_df), len(filtered)))
    assert len(filtered) == 3
    
    # Test cleaning
    cleaned = cleaner.clean_all_narratives(filtered, 'Consumer complaint narrative')
    print("  Clean test: 'cleaned_Consumer complaint narrative' column created")
    assert 'cleaned_Consumer complaint narrative' in cleaned.columns
    
    # Test analysis
    summary = analyzer.analyze_product_distribution(filtered, 'Product')
    print("  Analysis test: Product distribution analyzed")
    assert 'counts' in summary
    
    print("\n" + "="*60)
    print("✓ ALL FIXES VERIFIED SUCCESSFULLY")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
