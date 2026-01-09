"""Lightweight embedding generator using Hugging Face transformers without heavy dependencies."""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import gc
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    device: Optional[str] = None


class LightweightEmbeddingGenerator:
    """Generates embeddings using Hugging Face transformers with minimal dependencies."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding generator.
        
        Args:
            config: Configuration for embedding generation
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing embedding generator with config: {self.config}")
    
    def load_model(self):
        """Lazy load the model when needed."""
        if self.model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                
                logger.info(f"Loading model: {self.config.model_name}")
                
                # Set device
                device = self.config.device
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModel.from_pretrained(self.config.model_name)
                self.model.to(device)
                self.model.eval()
                
                logger.info(f"Model loaded on device: {device}")
                logger.info(f"Model dimensions: {self.model.config.hidden_size}")
                
            except ImportError as e:
                logger.error(f"Failed to import transformers/torch: {e}")
                logger.info("Falling back to numpy-based embeddings...")
                self.model = "numpy_fallback"
    
    def generate_embeddings_fallback(self, texts: List[str]) -> np.ndarray:
        """Generate simple embeddings as fallback when transformers not available.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        logger.warning("Using numpy fallback embeddings (not semantic)")
        
        # Simple TF-IDF like embedding (just for demonstration)
        # In production, you should use proper sentence transformers
        all_words = set()
        for text in texts:
            if isinstance(text, str):
                words = text.lower().split()
                all_words.update(words[:100])  # Limit vocabulary size
        
        vocab = list(all_words)
        if len(vocab) > 10000:  # Limit vocab size
            vocab = vocab[:10000]
        
        embeddings = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                continue
            
            words = text.lower().split()
            for word in words[:100]:  # Limit words per text
                if word in vocab:
                    idx = vocab.index(word)
                    embeddings[i, idx] += 1.0
        
        # Normalize
        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        
        return embeddings
    
    def generate_embeddings_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using transformers.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        import torch
        from transformers import AutoModel, AutoTokenizer
        
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        if self.model == "numpy_fallback":
            return self.generate_embeddings_fallback(texts)
        
        device = next(self.model.parameters()).device
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use mean pooling
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            embeddings = embeddings.cpu().numpy()
        
        return embeddings
    
    def generate_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([])
        
        batch_size = batch_size or self.config.batch_size
        logger.info(f"Generating embeddings for {len(texts):,} texts (batch_size={batch_size})")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                if self.model is None:
                    self.load_model()
                
                if isinstance(self.model, str) and self.model == "numpy_fallback":
                    batch_embeddings = self.generate_embeddings_fallback(batch_texts)
                else:
                    batch_embeddings = self.generate_embeddings_transformers(batch_texts)
                
                all_embeddings.append(batch_embeddings)
                
                if (i // batch_size) % 10 == 0:
                    logger.info(f"  Processed {min(i + batch_size, len(texts)):,}/{len(texts):,} texts")
            
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                # Fall back to numpy embeddings for this batch
                batch_embeddings = self.generate_embeddings_fallback(batch_texts)
                all_embeddings.append(batch_embeddings)
        
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated {len(embeddings):,} embeddings")
            logger.info(f"Embedding shape: {embeddings.shape}")
            logger.info(f"Embedding dtype: {embeddings.dtype}")
            return embeddings
        else:
            logger.error("No embeddings generated")
            return np.array([])
    
    def generate_embeddings_for_dataframe(self, df: pd.DataFrame, text_col: str = 'text',
                                          batch_size: Optional[int] = None) -> pd.DataFrame:
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
        embeddings = self.generate_embeddings(texts, batch_size)
        
        if len(embeddings) == 0:
            logger.error("Failed to generate embeddings")
            return df
        
        # Add embeddings to DataFrame
        result_df = df.copy()
        
        # Store embeddings as lists for parquet compatibility
        result_df['embeddings'] = [emb.tolist() for emb in embeddings]
        result_df['embedding_dim'] = embeddings.shape[1]
        
        logger.info(f"Added embeddings to DataFrame")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Memory usage: {embeddings.nbytes / 1024**2:.1f} MB")
        
        return result_df
    
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
        
        # Save to parquet
        save_df.to_parquet(output_path, index=False)
        
        logger.info(f"Saved embeddings to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / 1024**2:.1f} MB")
        logger.info(f"Columns saved: {list(save_df.columns)}")
        
        # Also save a sample for verification
        sample_path = output_path.with_suffix('.sample.json')
        sample_data = []
        for i, row in save_df.head(5).iterrows():
            sample_data.append({
                'chunk_id': row.get('chunk_id', ''),
                'product_category': row.get('product_category', ''),
                'text_preview': row.get('text', '')[:100] if 'text' in row else '',
                'embedding_shape': [len(row['embeddings'])] if 'embeddings' in row else []
            })
        
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Saved sample to {sample_path}")


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
        
        # Calculate embedding statistics from first few embeddings
        try:
            sample_embeddings = np.array(df['embeddings'].head(100).tolist())
            validation['statistics']['sample_mean'] = float(sample_embeddings.mean())
            validation['statistics']['sample_std'] = float(sample_embeddings.std())
        except:
            pass  # Skip statistics if can't compute
    else:
        validation['valid'] = False
        validation['errors'].append("Embeddings not in list/array format")
    
    return validation
