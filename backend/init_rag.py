"""Initialize RAG knowledge base - builds chunk store from raw papers."""
import logging
import sys
from pathlib import Path

from backend.core.config import settings
from backend.services.paper_processor import PaperProcessor

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_rag():
    """
    Build the RAG knowledge base:
    1. Load all PDFs from data/raw_papers/
    2. Chunk them
    3. Store in data/chunks.jsonl
    4. Track in data/papers_manifest.json
    
    Idempotent - can be run multiple times to refresh.
    """
    
    print("\n" + "="*70)
    print("INITIALIZING RAG KNOWLEDGE BASE")
    print("="*70)
    
    # Check if papers exist
    pdf_files = list(settings.raw_papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n No PDFs found in {settings.raw_papers_dir}")
        print("Add PDF files first!")
        return False
    
    print(f"\n✓ Found {len(pdf_files)} PDFs to process")
    print(f"  Papers directory: {settings.raw_papers_dir}")
    print(f"  Output chunks: {settings.chunks_file}")
    print(f"  Output manifest: {settings.manifest_path}")
    
    print(f"\nChunking config:")
    print(f"  Size: {settings.chunk_size}")
    print(f"  Overlap: {settings.chunk_overlap}")
    
    # Initialize processor
    processor = PaperProcessor(
        papers_dir=settings.raw_papers_dir,
        manifest_path=settings.manifest_path,
        chunks_file=settings.chunks_file,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    print("\n" + "-"*70)
    print("Processing papers...")
    print("-"*70)
    
    # Run initialization
    result = processor.process_all()
    
    # Report results
    print("\n" + "="*70)
    print("INITIALIZATION COMPLETE")
    print("="*70)
    
    print(f"\n✓ New papers processed: {result['new_processed']}")
    print(f"✓ Updated papers processed: {result['updated_processed']}")
    print(f"✓ Total chunks created: {result['total_chunks']}")
    
    status = result['status']
    print(f"\nRegistry status:")
    print(f"  Total papers tracked: {status['total_papers']}")
    print(f"  Total chunks stored: {status['total_chunks']}")
    
    print(f"\nFiles created/updated:")
    print(f"  - {settings.chunks_file} ({result['total_chunks']} chunks)")
    print(f"  - {settings.manifest_path} ({status['total_papers']} papers)")
    
    print("\n RAG knowledge base ready!")
    print("\nNext steps:")
    print("  1. Generate embeddings from chunks")
    print("  2. Store in vector database (Qdrant/FAISS)")
    print("  3. Build retrieval pipeline")
    print("  4. Connect to LLM for answers")
    
    return True


if __name__ == "__main__":
    try:
        success = initialize_rag()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)
