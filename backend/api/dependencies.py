"""Dependency injection for FastAPI endpoints."""

import logging
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from backend.services.preset_registry import PresetRegistry
from backend.services.paper_processor import PaperProcessor
from backend.services.chunk_store import ChunkStore
from backend.services.vector_store import VectorStore
from backend.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Singleton container for all services."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._rag_pipeline = None
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Initializing service container...")
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_store_path = self.data_dir / "chunks.jsonl"
        self.vector_store_path = self.data_dir / "vector_store"
        self.papers_dir = self.data_dir / "raw_papers"
        self.manifest_path = self.data_dir / "papers_manifest.json"
        
        # Initialize services (lazy - RAGPipeline initialized on demand)
        try:
            # Chunk and vector store
            logger.info("  - Initializing chunk store...")
            self.chunk_store = ChunkStore(str(self.chunk_store_path))
            
            logger.info("  - Initializing vector store...")
            self.vector_store = VectorStore(persist_directory=str(self.vector_store_path), project_root=self.project_root)
            
            # Paper processor
            logger.info("  - Initializing paper processor...")
            self.paper_processor = PaperProcessor(
                papers_dir=self.papers_dir,
                manifest_path=self.manifest_path,
                chunks_file=self.chunk_store_path,
                chunk_size=1000,
                chunk_overlap=200,
                project_root=self.project_root,
            )
            
            # Embedding service
            logger.info("  - Initializing embedding service...")
            self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
            
            # Preset registry
            logger.info("  - Initializing preset registry...")
            self.preset_registry = PresetRegistry(project_root=self.project_root)
            
            logger.info("✓ Core services initialized successfully")
            logger.info("  Note: RAGPipeline will be initialized on first query (requires GROQ_API_KEY)")
            
        except Exception as e:
            logger.error(f"Failed to initialize core services: {str(e)}", exc_info=True)
            raise
        
        self._initialized = True
    
    def get_relative_path(self, path: Path) -> str:
        """Get relative path from project root for security/logging."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            # Fallback if path is not relative to project root
            return str(path)
    
    def get_relative_path(self, path: Path) -> str:
        """Get relative path from project root for security/logging."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            # Fallback if path is not relative to project root
            return str(path)
    
    def get_rag_pipeline(self):
        """Lazy-load RAGPipeline only when needed."""
        if self._rag_pipeline is None:
            logger.info("Initializing RAGPipeline on first use...")
            
            # Check if GROQ_API_KEY is set
            if not os.getenv("GROQ_API_KEY"):
                raise RuntimeError(
                    "GROQ_API_KEY environment variable not set. "
                    "Please set it in .env file or environment before querying."
                )
            
            try:
                from backend.services.rag_pipeline import RAGPipeline
                
                self._rag_pipeline = RAGPipeline(
                    chunk_store_path=str(self.chunk_store_path),
                    vector_store_path=str(self.vector_store_path),
                    preset_registry=self.preset_registry,
                )
                logger.info("✓ RAGPipeline initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAGPipeline: {str(e)}", exc_info=True)
                raise
        
        return self._rag_pipeline


@lru_cache(maxsize=1)
def get_service_container() -> ServiceContainer:
    """Get or create the service container (singleton)."""
    return ServiceContainer()


# Dependency functions for FastAPI

async def get_rag_pipeline():
    """Get RAG pipeline for a request."""
    container = get_service_container()
    return container.get_rag_pipeline()


async def get_preset_registry() -> PresetRegistry:
    """Get preset registry for a request."""
    container = get_service_container()
    return container.preset_registry


async def get_paper_processor() -> PaperProcessor:
    """Get paper processor for a request."""
    container = get_service_container()
    return container.paper_processor


async def get_chunk_store() -> ChunkStore:
    """Get chunk store for a request."""
    container = get_service_container()
    return container.chunk_store


async def get_vector_store() -> VectorStore:
    """Get vector store for a request."""
    container = get_service_container()
    return container.vector_store


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service for a request."""
    container = get_service_container()
    return container.embedding_service
