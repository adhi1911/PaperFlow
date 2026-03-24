"""Orchestrate paper processing: load → chunk → store → track."""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from backend.services.paper_registry import PaperRegistry
from backend.services.document_loader import DocumentLoader
from backend.services.chunking_service import ChunkingService
from backend.services.chunk_store import ChunkStore


logger = logging.getLogger(__name__)


class PaperProcessor:
    """End-to-end: detect new/updated papers, chunk, store, and track."""

    def __init__(
        self,
        papers_dir: Path,
        manifest_path: Path,
        chunks_file: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        project_root: Optional[Path] = None,
    ):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)

        self.registry = PaperRegistry(manifest_path)
        self.loader = DocumentLoader(self.papers_dir, project_root=project_root)
        self.chunker = ChunkingService(chunk_size, chunk_overlap)
        self.store = ChunkStore(chunks_file)

    def process_all(self) -> Dict[str, Any]:
        """Process new and updated papers. Returns summary."""
        new = self.registry.get_new_papers(self.papers_dir)
        updated = self.registry.get_updated_papers(self.papers_dir)

        logger.info(f"New: {len(new)}, Updated: {len(updated)}")

        # Re-process updated papers (remove old chunks)
        for paper_path in updated:
            logger.info(f"Re-processing: {paper_path.name}")
            self.store.remove_by_source(paper_path.name)
            self._add_paper(paper_path)

        # Process new papers
        for paper_path in new:
            logger.info(f"Processing: {paper_path.name}")
            self._add_paper(paper_path)

        return {
            "new_processed": len(new),
            "updated_processed": len(updated),
            "total_chunks": self.store.get_stats()["total_chunks"],
            "status": self.registry.get_status(),
        }

    def _add_paper(self, paper_path: Path) -> None:
        """Load, chunk, store, and register a paper."""
        # Load single paper
        docs = self.loader.load_specific_pdf(paper_path.name)
        if not docs:
            logger.warning(f"Failed to load: {paper_path.name}")
            return

        # Chunk
        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            logger.warning(f"No chunks from: {paper_path.name}")
            return

        # Store
        self.store.append_chunks(chunks)

        # Register
        file_hash = PaperRegistry._get_file_hash(paper_path)
        self.registry.register_paper(paper_path.name, file_hash, len(chunks))

        logger.info(f"Processed {paper_path.name}: {len(chunks)} chunks")

    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        stats = self.store.get_stats()
        return {
            "registry": self.registry.get_status(),
            "chunks": stats,
        }
