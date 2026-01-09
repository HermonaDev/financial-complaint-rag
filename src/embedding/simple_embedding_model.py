import numpy as np
import zlib

class SimpleEmbeddingModel:
    """
    A simple deterministic embedding model for compatibility.
    Produces 384-dimensional vectors based on input text hashing.
    """
    def __init__(self, dimension=384):
        self.dimension = dimension

    def encode(self, text):
        """
        Generates a deterministic embedding vector for the input text.
        """
        if not text:
            return np.zeros(self.dimension, dtype=np.float32)
            
        # Use checksum of text to seed random generator for determinism
        seed = zlib.adler32(text.encode('utf-8')) & 0xffffffff
        rng = np.random.RandomState(seed)
        
        # Generate random vector
        vector = rng.rand(self.dimension).astype(np.float32)
        
        # Normalize vector (L2 norm)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector

    def encode_batch(self, texts):
        """
        Encodes a list of texts.
        """
        return np.array([self.encode(t) for t in texts])
