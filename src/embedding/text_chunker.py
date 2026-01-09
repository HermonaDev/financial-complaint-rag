"""Text chunking utilities for complaint narratives."""

from typing import List, Dict, Any
import re
from dataclasses import dataclass
import logging
import pandas as pd  # Import pandas at the top

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_index: int
    total_chunks: int
    complaint_id: str
    product_category: str
    start_char: int
    end_char: int


class TextChunker:
    """Chunks text into overlapping segments."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize chunker with size and overlap parameters.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        logger.info(f"Initialized chunker: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, complaint_id: str, product_category: str) -> List[TextChunk]:
        """Chunk a single text into overlapping segments.
        
        Args:
            text: Text to chunk
            complaint_id: ID of the complaint
            product_category: Product category
            
        Returns:
            List of TextChunk objects
        """
        if not text or not isinstance(text, str):
            return []
        
        text_length = len(text)
        
        # If text is shorter than chunk_size, return single chunk
        if text_length <= self.chunk_size:
            return [TextChunk(
                text=text,
                chunk_index=0,
                total_chunks=1,
                complaint_id=complaint_id,
                product_category=product_category,
                start_char=0,
                end_char=text_length
            )]
        
        # Calculate chunk positions
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # Ensure we don't break in the middle of a word if possible
            if end < text_length:
                # Look for the last space in the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start and (last_space - start) > (self.chunk_size * 0.7):
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    complaint_id=complaint_id,
                    product_category=product_category,
                    start_char=start,
                    end_char=end
                ))
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start <= chunks[-1].start_char if chunks else 0:
                start = chunks[-1].end_char if chunks else 0
        
        # Update total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks
        
        logger.debug(f"Chunked text of length {text_length} into {total_chunks} chunks")
        
        return chunks
    
    def chunk_dataframe(self, df, text_col: str = 'cleaned_narrative', 
                        id_col: str = 'Complaint ID', 
                        product_col: str = 'Product') -> List[TextChunk]:
        """Chunk all texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_col: Column with text to chunk
            id_col: Column with complaint IDs
            product_col: Column with product categories
            
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        total_texts = len(df)
        
        logger.info(f"Starting chunking of {total_texts:,} texts...")
        
        for idx, row in df.iterrows():
            text = row.get(text_col, '')
            complaint_id = str(row.get(id_col, f'unknown_{idx}'))
            product_category = row.get(product_col, 'Unknown')
            
            chunks = self.chunk_text(text, complaint_id, product_category)
            all_chunks.extend(chunks)
            
            # Log progress
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Chunked {idx + 1:,}/{total_texts:,} texts "
                           f"({len(all_chunks):,} chunks so far)")
        
        logger.info(f"Completed chunking: {len(all_chunks):,} total chunks "
                   f"from {total_texts:,} texts")
        logger.info(f"Average chunks per text: {len(all_chunks)/total_texts:.2f}")
        
        return all_chunks
    
    def chunks_to_dataframe(self, chunks: List[TextChunk]) -> pd.DataFrame:
        """Convert list of TextChunk objects to DataFrame.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            DataFrame with chunk data
        """
        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                'chunk_id': f"{chunk.complaint_id}_{chunk.chunk_index}",
                'complaint_id': chunk.complaint_id,
                'product_category': chunk.product_category,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'text': chunk.text,
                'text_length': len(chunk.text)
            })
        
        df = pd.DataFrame(chunk_dicts)
        
        # Add statistics
        if len(df) > 0:
            logger.info(f"Created DataFrame with {len(df):,} chunks")
            logger.info(f"Average chunk length: {df['text_length'].mean():.0f} characters")
            logger.info(f"Chunk length distribution:")
            logger.info(f"  Min: {df['text_length'].min()} chars")
            logger.info(f"  Max: {df['text_length'].max()} chars")
            logger.info(f"  Mean: {df['text_length'].mean():.0f} chars")
            logger.info(f"  Median: {df['text_length'].median():.0f} chars")
        
        return df