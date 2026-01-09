import faiss
import pickle
import numpy as np
import os
from src.embedding.simple_embedding_model import SimpleEmbeddingModel

class RAGRetriever:
    """
    Retrieves relevant chunks using FAISS and a simple embedding model.
    """
    def __init__(self, vector_store_path="vector_store/faiss_prebuilt.index", metadata_path="vector_store/metadata_prebuilt.pkl"):
        self.vector_store_path = vector_store_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        self.embedder = SimpleEmbeddingModel()
        self._load_resources()

    def _load_resources(self):
        """Loads the FAISS index and metadata."""
        if not os.path.exists(self.vector_store_path):
            raise FileNotFoundError(f"Vector store not found at {self.vector_store_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        print(f"Loading FAISS index from {self.vector_store_path}...")
        self.index = faiss.read_index(self.vector_store_path)
        
        print(f"Loading metadata from {self.metadata_path}...")
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        print("Resources loaded successfully.")

    def search(self, query, k=5):
        """
        Searches for the top-k most relevant chunks for the query.
        """
        query_vector = self.embedder.encode(query)
        # FAISS expects a 2D array (batch_size, dimension)
        query_vector = np.array([query_vector]).astype(np.float32)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata) and idx >= 0:
                item = self.metadata[idx].copy()
                item['score'] = float(distances[0][i])
                results.append(item)
                
        return results
