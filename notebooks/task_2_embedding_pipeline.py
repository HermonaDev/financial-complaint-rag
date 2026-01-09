#!/usr/bin/env python3
"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
Main pipeline for creating embeddings from stratified sample.
"""

import sys
from pathlib import Path
import logging
import pandas as pd  # Import pandas at the top

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sampling.stratified_sampler import StratifiedSampler, load_complaints_data, create_sample_report
from embedding.text_chunker import TextChunker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task_2_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main Task 2 execution pipeline."""
    logger.info("="*60)
    logger.info("STARTING TASK 2: Text Chunking and Embedding Pipeline")
    logger.info("="*60)
    
    try:
        # Step 1: Load filtered data
        logger.info("\nSTEP 1: Loading filtered complaint data...")
        input_path = Path('data/processed/filtered_complaints.csv')
        if not input_path.exists():
            logger.error(f"Filtered data not found: {input_path}")
            return
        
        df = load_complaints_data(input_path)
        logger.info(f"Loaded {len(df):,} filtered complaints")
        
        # Step 2: Create stratified sample
        logger.info("\nSTEP 2: Creating stratified sample...")
        sampler = StratifiedSampler(target_sample_size=12000, random_state=42)
        sampled_df = sampler.sample_complaints(df, product_col='Product')
        
        if len(sampled_df) == 0:
            logger.error("Failed to create sample")
            return
        
        # Save sample
        sample_path = Path('data/processed/sampled_complaints.csv')
        sampler.save_sample(sampled_df, sample_path)
        
        # Create sample report
        report = create_sample_report(sampled_df, df)
        logger.info(f"\nSample Report:")
        logger.info(f"  Sample size: {report['sample_size']:,}")
        logger.info(f"  Sampling rate: {report['sampling_rate']:.2%}")
        
        # Step 3: Chunk narratives
        logger.info("\nSTEP 3: Chunking complaint narratives...")
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_dataframe(
            sampled_df, 
            text_col='cleaned_narrative',
            id_col='Complaint ID',
            product_col='Product'
        )
        
        # Convert chunks to DataFrame
        chunks_df = chunker.chunks_to_dataframe(chunks)
        
        # Save chunks
        chunks_path = Path('data/processed/complaint_chunks.csv')
        chunks_df.to_csv(chunks_path, index=False)
        logger.info(f"Saved {len(chunks_df):,} chunks to {chunks_path}")
        
        # Step 4: Generate summary
        logger.info("\nSTEP 4: Generating pipeline summary...")
        summary = {
            'total_complaints_loaded': len(df),
            'sample_size': len(sampled_df),
            'total_chunks_generated': len(chunks_df),
            'avg_chunks_per_complaint': len(chunks_df) / len(sampled_df),
            'chunk_statistics': {
                'avg_length': float(chunks_df['text_length'].mean()),
                'min_length': int(chunks_df['text_length'].min()),
                'max_length': int(chunks_df['text_length'].max()),
                'median_length': float(chunks_df['text_length'].median())
            },
            'file_paths': {
                'filtered_data': str(input_path),
                'sample_data': str(sample_path),
                'chunks_data': str(chunks_path)
            }
        }
        
        # Print final summary
        print("\n" + "="*60)
        print("TASK 2 PIPELINE COMPLETE - PHASE 1")
        print("="*60)
        print(f"Loaded: {summary['total_complaints_loaded']:,} filtered complaints")
        print(f"Sampled: {summary['sample_size']:,} complaints")
        print(f"Generated: {summary['total_chunks_generated']:,} text chunks")
        print(f"Average chunks per complaint: {summary['avg_chunks_per_complaint']:.2f}")
        print(f"Average chunk length: {summary['chunk_statistics']['avg_length']:.0f} chars")
        print(f"Chunk length range: {summary['chunk_statistics']['min_length']} to "
              f"{summary['chunk_statistics']['max_length']} chars")
        print("\nNext: Run embedding generation and vector store creation")
        print("="*60)
        
        logger.info("Task 2 Phase 1 completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Task 2 pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
