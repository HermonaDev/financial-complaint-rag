"""Embedding modules for text chunking and embedding generation."""

from .text_chunker import TextChunker

# Import lightweight generator only (avoid sentence-transformers imports)
try:
    from .lightweight_embedding_generator import LightweightEmbeddingGenerator, EmbeddingConfig, validate_embeddings
    __all__ = ["TextChunker", "LightweightEmbeddingGenerator", "EmbeddingConfig", "validate_embeddings"]
except ImportError as e:
    # If imports fail, only export TextChunker
    __all__ = ["TextChunker"]
    print(f"Note: LightweightEmbeddingGenerator not available: {e}")
