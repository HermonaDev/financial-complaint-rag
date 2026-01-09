"""Embedding generation for complaint text chunks."""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import gc

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence-transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        """Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"Model loaded successfully")
            logger.info(f"Model dimensions: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32,
                            show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts):,} texts...")
        logger.info(f"Batch size: {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            logger.info(f"Generated {len(embeddings):,} embeddings")
            logger.info(f"Embedding shape: {embeddings.shape}")
            logger.info(f"Embedding dtype: {embeddings.dtype}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embeddings_for_dataframe(self, df: pd.DataFrame, text_col: str = 'text',
                                          batch_size: int = 32) -> pd.DataFrame:
        """Generate embeddings for text chunks in a DataFrame.
        
        Args:
            df: DataFrame containing text chunks
            text_col: Column containing text to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            DataFrame with added embedding column
        """
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in DataFrame")
        
        logger.info(f"Generating embeddings for DataFrame with {len(df):,} rows")
        
        # Extract texts
        texts = df[text_col].tolist()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Add embeddings to DataFrame
        df = df.copy()
        df['embeddings'] = list(embeddings)
        
        # Calculate embedding statistics
        embedding_dim = embeddings.shape[1]
        logger.info(f"Added embeddings to DataFrame")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info(f"Memory usage: {embeddings.nbytes / 1024**2:.1f} MB")
        
        return df
    
    def save_embeddings(self, df: pd.DataFrame, output_path: Path,
                        include_text: bool = True):
        """Save embeddings and metadata to parquet file.
        
        Args:
            df: DataFrame with embeddings
            output_path: Path to save embeddings
            include_text: Whether to include text in output
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_df = df.copy()
        
        if not include_text and 'text' in save_df.columns:
            save_df = save_df.drop(columns=['text'])
        
        # Convert embeddings to list for parquet compatibility
        if 'embeddings' in save_df.columns:
            save_df['embeddings'] = save_df['embeddings'].apply(list)
        
        # Save to parquet
        save_df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024**2:.1f} MB")
        logger.info(f"Columns saved: {list(save_df.columns)}")
    
    def load_embeddings(self, input_path: Path) -> pd.DataFrame:
        """Load embeddings from parquet file.
        
        Args:
            input_path: Path to embeddings file
            
        Returns:
            DataFrame with embeddings
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {input_path}")
        
        logger.info(f"Loading embeddings from {input_path}")
        df = pd.read_parquet(input_path)
        
        # Convert embeddings back to numpy arrays
        if 'embeddings' in df.columns:
            df['embeddings'] = df['embeddings'].apply(np.array)
        
        logger.info(f"Loaded {len(df):,} embeddings")
        
        return df


def validate_embeddings(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate embeddings in DataFrame.
    
    Args:
        df: DataFrame with embeddings
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'errors': [],
        'statistics': {}
    }
    
    if 'embeddings' not in df.columns:
        validation['valid'] = False
        validation['errors'].append("'embeddings' column not found")
        return validation
    
    # Check embedding dimensions
    sample_embedding = df['embeddings'].iloc[0]
    if isinstance(sample_embedding, (list, np.ndarray)):
        embedding_dim = len(sample_embedding)
        validation['statistics']['embedding_dimension'] = embedding_dim
        validation['statistics']['num_embeddings'] = len(df)
        
        # Check all embeddings have same dimension
        dimensions = df['embeddings'].apply(len)
        if not dimensions.nunique() == 1:
            validation['valid'] = False
            validation['errors'].append("Inconsistent embedding dimensions")
        
        # Calculate embedding statistics
        all_embeddings = np.vstack(df['embeddings'].values)
        validation['statistics']['embedding_mean'] = float(all_embeddings.mean())
        validation['statistics']['embedding_std'] = float(all_embeddings.std())
        validation['statistics']['memory_mb'] = all_embeddings.nbytes / 1024**2
    else:
        validation['valid'] = False
        validation['errors'].append("Embeddings not in list/array format")
    
    return validation
