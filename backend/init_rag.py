"""Initialize RAG: Load papers → Chunk → Embed → Store in vector DB."""
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from backend.services.paper_processor import PaperProcessor
from backend.services.chunk_store import ChunkStore
from backend.services.embedding import EmbeddingService
from backend.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Suppress verbose logs from HuggingFace Hub
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def main():
    """Initialize RAG knowledge base with embeddings and vector store."""
    
    print("\n" + "=" * 70)
    print("PaperFlow RAG Initialization")
    print("=" * 70)
    
    # Ensure data directory exists
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    # ===== Step 1: Process papers (load → chunk → store) =====
    print("\nStep 1: Processing papers...")
    print("-" * 70)
    
    try:
        processor = PaperProcessor(
            papers_dir=data_dir / "raw_papers",
            manifest_path=data_dir / "papers_manifest.json",
            chunks_file=data_dir / "chunks.jsonl",
            project_root=project_root
        )
        result = processor.process_all()
        
        print(f"[OK] {result['new_processed']} new papers processed")
        print(f"[OK] {result['updated_processed']} papers reprocessed")
        print(f"[OK] Total chunks: {result['total_chunks']}")
    except Exception as e:
        print(f"[ERROR] Paper processing failed: {e}")
        logger.error(f"Paper processing error: {e}", exc_info=True)
        return False
    
    # ===== Step 2: Generate embeddings =====
    print("\nStep 2: Generating embeddings...")
    print("-" * 70)
    
    try:
        # Load all chunks
        store = ChunkStore("data/chunks.jsonl")
        all_chunks = store.load_all()
        
        if not all_chunks:
            print("[ERROR] No chunks found. Please process papers first.")
            return False
        
        # Initialize embedding service
        embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        print(f"[OK] {embedding_service}")
        
        # Generate embeddings
        print(f"Embedding {len(all_chunks)} chunks...")
        embedding_results = embedding_service.embed_chunks(all_chunks, batch_size=32)
        print(f"[OK] Generated {len(embedding_results)} embeddings")
    except Exception as e:
        print(f"[ERROR] Embedding generation failed: {e}")
        logger.error(f"Embedding generation error: {e}", exc_info=True)
        return False
    
    # ===== Step 3: Store embeddings =====
    print("\nStep 3: Storing embeddings in vector database...")
    print("-" * 70)
    
    try:
        vector_store = VectorStore(persist_directory=data_dir / "vector_store", project_root=project_root)
        added = vector_store.add_embeddings(embedding_results, auto_delete_source=True)
        print(f"[OK] Added {added} embeddings to vector store")
    except Exception as e:
        print(f"[ERROR] Vector store indexing failed: {e}")
        logger.error(f"Vector store error: {e}", exc_info=True)
        return False
    
    # ===== Summary =====
    elapsed = time.time() - start_time
    chunk_stats = store.get_stats()
    vector_stats = vector_store.get_stats()
    
    print("\n" + "=" * 70)
    print("Initialization Complete!")
    print("=" * 70)
    
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Chunks: {chunk_stats['total_chunks']} from {chunk_stats['total_sources']} sources")
    print(f"Vectors: {vector_stats['total_vectors']} in vector store")
    print(f"Embedding dimension: {embedding_service.get_embedding_dim()}")
    
    print(f"\nFiles created:")
    print(f"   - data/chunks.jsonl ({chunk_stats['total_chunks']} chunks)")
    print(f"   - data/papers_manifest.json")
    print(f"   - data/vector_store/ (Chroma DB)")
    
    print("\nReady for retrieval & generation!")
    print("   Next: Use VectorStore.search() for semantic search\n")
    
    return True


if __name__ == "__main__":
    import sys
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)
