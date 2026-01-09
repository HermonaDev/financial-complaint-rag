#!/usr/bin/env python3
"""
Task 2 Phase 2: Embedding Generation and Vector Store Creation
Generate embeddings and create FAISS vector store.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedding.embedding_generator import EmbeddingGenerator, validate_embeddings
from vector_store.faiss_store import FAISSVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task_2_phase2.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main Phase 2 execution pipeline."""
    logger.info("="*60)
    logger.info("TASK 2 PHASE 2: Embedding Generation and Vector Store")
    logger.info("="*60)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Step 1: Load chunked data
        logger.info("\nSTEP 1: Loading chunked data...")
        chunks_path = Path('data/processed/complaint_chunks.csv')
        if not chunks_path.exists():
            logger.error(f"Chunked data not found: {chunks_path}")
            return
        
        chunks_df = pd.read_csv(chunks_path)
        logger.info(f"Loaded {len(chunks_df):,} text chunks")
        logger.info(f"Sample chunk: {chunks_df['text'].iloc[0][:100]}...")
        
        # Step 2: Generate embeddings
        logger.info("\nSTEP 2: Generating embeddings...")
        generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=None  # Auto-detect
        )
        
        # Generate embeddings
        logger.info("Starting embedding generation (this may take a while)...")
        embeddings_df = generator.generate_embeddings_for_dataframe(
            chunks_df,
            text_col='text',
            batch_size=32
        )
        
        # Validate embeddings
        validation = validate_embeddings(embeddings_df)
        logger.info("\nEmbedding validation:")
        logger.info(f"  Valid: {validation['valid']}")
        logger.info(f"  Dimension: {validation['statistics'].get('embedding_dimension', 'N/A')}")
        logger.info(f"  Memory: {validation['statistics'].get('memory_mb', 0):.1f} MB")
        
        if not validation['valid']:
            logger.error(f"Embedding validation failed: {validation['errors']}")
            return
        
        # Step 3: Save embeddings
        logger.info("\nSTEP 3: Saving embeddings...")
        embeddings_path = Path('data/processed/complaint_embeddings.parquet')
        generator.save_embeddings(embeddings_df, embeddings_path, include_text=True)
        
        # Step 4: Create vector store
        logger.info("\nSTEP 4: Creating FAISS vector store...")
        
        # Prepare embeddings array
        embeddings_array = np.vstack(embeddings_df['embeddings'].values)
        logger.info(f"Embeddings array shape: {embeddings_array.shape}")
        
        # Prepare metadata
        metadata = []
        for _, row in embeddings_df.iterrows():
            metadata.append({
                'chunk_id': row.get('chunk_id', ''),
                'complaint_id': row.get('complaint_id', ''),
                'product_category': row.get('product_category', ''),
                'chunk_index': int(row.get('chunk_index', 0)),
                'text': row.get('text', '')[:200]  # Store first 200 chars
            })
        
        # Create FAISS index
        vector_store = FAISSVectorStore(dimension=embeddings_array.shape[1], index_type="Flat")
        vector_store.create_index(embeddings_array, metadata)
        
        # Save vector store
        vector_store_path = Path('vector_store/faiss_index')
        vector_store.save(vector_store_path)
        
        # Get statistics
        stats = vector_store.get_stats()
        logger.info("\nVector store statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Step 5: Test the vector store
        logger.info("\nSTEP 5: Testing vector store...")
        
        # Test with a sample query
        test_queries = [
            "credit card fraud",
            "bank account fees",
            "money transfer delay",
            "loan payment issue"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            query_embedding = generator.model.encode([query])
            results = vector_store.search(query_embedding, k=3)
            
            for i, result in enumerate(results):
                metadata = result['metadata']
                logger.info(f"  Result {i+1}: Score={result['score']:.3f}, "
                          f"Product={metadata.get('product_category', 'N/A')}, "
                          f"Text={metadata.get('text', '')[:80]}...")
        
        # Step 6: Generate final summary
        logger.info("\nSTEP 6: Generating final summary...")
        
        summary = {
            'phase': 'Task 2 Phase 2 Complete',
            'chunks_processed': len(chunks_df),
            'embeddings_generated': len(embeddings_df),
            'embedding_dimension': embeddings_array.shape[1],
            'vector_store_size': vector_store.index.ntotal if vector_store.index else 0,
            'files_created': {
                'embeddings': str(embeddings_path),
                'faiss_index': str(vector_store_path.with_suffix('.faiss')),
                'metadata': str(vector_store_path.with_suffix('.metadata.pkl'))
            }
        }
        
        # Print final summary
        print("\n" + "="*60)
        print("TASK 2 PHASE 2 COMPLETE")
        print("="*60)
        print(f"Processed: {summary['chunks_processed']:,} text chunks")
        print(f"Generated: {summary['embeddings_generated']:,} embeddings")
        print(f"Embedding dimension: {summary['embedding_dimension']}")
        print(f"Vector store size: {summary['vector_store_size']:,} vectors")
        print(f"\nFiles created:")
        print(f"  Embeddings: {summary['files_created']['embeddings']}")
        print(f"  FAISS index: {summary['files_created']['faiss_index']}")
        print(f"  Metadata: {summary['files_created']['metadata']}")
        print("\n" + "="*60)
        print("✅ TASK 2 COMPLETE!")
        print("="*60)
        
        logger.info("Task 2 Phase 2 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Phase 2 pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
