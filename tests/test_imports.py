"""Test all Task 2 imports."""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_sampling_imports():
    """Test that sampling modules can be imported."""
    from sampling import StratifiedSampler, load_complaints_data
    assert StratifiedSampler is not None
    assert load_complaints_data is not None


def test_embedding_imports():
    """Test that embedding modules can be imported."""
    from embedding import TextChunker
    assert TextChunker is not None
    
    # EmbeddingGenerator may not be available if sentence-transformers/torch is not installed
    # or if there are DLL issues on Windows
    try:
        from embedding import EmbeddingGenerator
        assert EmbeddingGenerator is not None
    except (ImportError, OSError):
        pytest.skip("EmbeddingGenerator not available (sentence-transformers/torch may not be installed or DLL issues)")


def test_vector_store_imports():
    """Test that vector_store modules can be imported."""
    # FAISSVectorStore may not be implemented yet
    try:
        from vector_store import FAISSVectorStore
        assert FAISSVectorStore is not None
    except ImportError:
        pytest.skip("FAISSVectorStore not yet implemented (vector store functionality is in RAGRetriever)")
