"""Persist LangChain Document chunks to JSONL."""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class ChunkStore:
    """Save and load chunks to/from JSONL file (one Document per line)."""

    def __init__(self, chunks_file: Path):
        self.chunks_file = Path(chunks_file)
        self.chunks_file.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(self, doc: Document) -> Dict[str, Any]:
        """Convert Document to serializable dict."""
        return {"page_content": doc.page_content, "metadata": doc.metadata}

    def _from_dict(self, data: Dict[str, Any]) -> Document:
        """Convert dict to Document."""
        return Document(
            page_content=data["page_content"],
            metadata=data.get("metadata", {})
        )

    def append_chunks(self, chunks: List[Document]) -> None:
        """Append chunks to JSONL file."""
        try:
            with open(self.chunks_file, "a", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(self._to_dict(chunk)) + "\n")
            logger.info(f"Appended {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error appending chunks: {e}")
            raise

    def remove_by_source(self, source: str) -> int:
        """Remove all chunks from a source. Returns count removed."""
        chunks = self.load_all()
        original = len(chunks)
        kept = [c for c in chunks if c.metadata.get("source") != source]
        removed = original - len(kept)
        
        if removed > 0:
            self._save_all(kept)
            logger.info(f"Removed {removed} chunks from {source}")
        
        return removed

    def load_all(self) -> List[Document]:
        """Load all chunks from JSONL."""
        if not self.chunks_file.exists():
            return []

        chunks = []
        try:
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        chunks.append(self._from_dict(data))
            logger.debug(f"Loaded {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
        
        return chunks

    def _save_all(self, chunks: List[Document]) -> None:
        """Overwrite JSONL with chunks."""
        try:
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(self._to_dict(chunk)) + "\n")
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            raise

    def get_by_source(self, source: str) -> List[Document]:
        """Get all chunks from a source."""
        return [c for c in self.load_all() if c.metadata.get("source") == source]

    def get_stats(self) -> Dict[str, Any]:
        """Get chunk statistics."""
        chunks = self.load_all()
        sources = list(set(c.metadata.get("source") for c in chunks if "source" in c.metadata))
        
        return {
            "total_chunks": len(chunks),
            "total_sources": len(sources),
            "chunks_per_source": {s: len(self.get_by_source(s)) for s in sorted(sources)}
        }
