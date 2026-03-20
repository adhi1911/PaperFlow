"""Track processed papers using file hashes."""
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PaperRegistry:
    """Track processed papers in JSON manifest (filename, hash, chunk count)."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_manifest()


    def _load_manifest(self):
        """Load manifest from disk or create empty."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.papers = {p["filename"]: p for p in data.get("papers", [])}
                logger.info(f"Loaded manifest: {len(self.papers)} papers")
            except Exception as e:
                logger.error(f"Error loading manifest: {e}")
                self.papers = {}
        else:
            self.papers = {}
            logger.info("Created new manifest")

    def _save_manifest(self):
        """Save manifest to disk."""
        data = {"papers": list(self.papers.values())}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved manifest: {len(self.papers)} papers")

    @staticmethod
    def _get_file_hash(file_path: Path) -> str:
        """Calculate MD5 hash to detect file changes."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def register_paper(self, filename: str, file_hash: str, num_chunks: int) -> None:
        """Record paper in manifest."""
        self.papers[filename] = {
            "filename": filename,
            "file_hash": file_hash,
            "num_chunks": num_chunks,
            "processed_at": datetime.utcnow().isoformat()
        }
        self._save_manifest()
        logger.info(f"Registered: {filename} ({num_chunks} chunks)")

    def get_paper_info(self, filename: str) -> Optional[Dict]:
        """Get info for specific paper."""
        return self.papers.get(filename)

    def is_paper_processed(self, filename: str) -> bool:
        """Check if paper is in manifest."""
        return filename in self.papers

    def get_new_papers(self, papers_dir: Path) -> List[Path]:
        """Get PDFs in folder not in manifest."""
        pdf_files = list(papers_dir.glob("*.pdf"))
        new_papers = [p for p in pdf_files if p.name not in self.papers]
        if new_papers:
            logger.info(f"Found {len(new_papers)} new papers")
        return new_papers

    def get_updated_papers(self, papers_dir: Path) -> List[Path]:
        """Get PDFs in manifest that have changed (hash mismatch)."""
        updated = []
        for filename, info in self.papers.items():
            file_path = papers_dir / filename
            if not file_path.exists():
                continue
            
            if self._get_file_hash(file_path) != info.get("file_hash"):
                updated.append(file_path)
                logger.info(f"Detected update: {filename}")
        
        return updated

    def get_all_papers(self) -> List[Dict]:
        """Get list of all tracked papers."""
        return list(self.papers.values())

    def get_total_chunks(self) -> int:
        """Get total chunks across all papers."""
        return sum(p["num_chunks"] for p in self.papers.values())

    def get_status(self) -> Dict:
        """Get comprehensive status of all papers."""
        return {
            "total_papers": len(self.papers),
            "total_chunks": self.get_total_chunks(),
            "papers": self.get_all_papers(),
        }