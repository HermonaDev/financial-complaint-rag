"""Embedding modules for text chunking and embedding generation."""

from .text_chunker import TextChunker

# Import embedding generators
__all__ = ["TextChunker"]

# Try to import lightweight generator (preferred for testing)
try:
    from .lightweight_embedding_generator import LightweightEmbeddingGenerator, EmbeddingConfig, validate_embeddings
    __all__.extend(["LightweightEmbeddingGenerator", "EmbeddingConfig", "validate_embeddings"])
except (ImportError, OSError) as e:
    # Silently skip if not available (e.g., missing dependencies or DLL issues)
    pass

# Try to import full embedding generator (requires sentence-transformers and torch)
try:
    from .embedding_generator import EmbeddingGenerator
    __all__.append("EmbeddingGenerator")
except (ImportError, OSError) as e:
    # Silently skip if not available (e.g., missing dependencies or DLL issues)
    pass
