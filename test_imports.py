#!/usr/bin/env python3
"""Test all Task 2 imports."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing Task 2 imports...")

try:
    # Test sampling imports
    from sampling import StratifiedSampler, load_complaints_data
    print("✅ sampling imports work")
    
    # Test embedding imports
    from embedding import TextChunker, EmbeddingGenerator
    print("✅ embedding imports work")
    
    # Test vector_store imports
    from vector_store import FAISSVectorStore
    print("✅ vector_store imports work")
    
    print("\n✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
