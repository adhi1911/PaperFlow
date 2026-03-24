"""Load PDFs from directory using LangChain."""
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from backend.utils.path_utils import get_relative_path


logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load PDFs and return LangChain Document objects."""

    def __init__(self, papers_dir: Path, project_root: Optional[Path] = None):
        self.papers_dir = Path(papers_dir)
        self.project_root = project_root
        self.papers_dir.mkdir(parents=True, exist_ok=True)

    def load_all_pdfs(self) -> List[Document]:
        """Load all PDFs from directory (optimized with DirectoryLoader)."""
        loader = DirectoryLoader(
            str(self.papers_dir),
            glob="*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True,
        )
        
        try:
            documents = loader.load()
            relative_path = get_relative_path(self.papers_dir, self.project_root)
            logger.info(f"Loaded {len(documents)} documents from {relative_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def load_specific_pdf(self, filename: str) -> List[Document]:
        """Load single PDF file."""
        file_path = self.papers_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {filename}")
            return []
        
        try:
            loader = PyMuPDFLoader(str(file_path))
            documents = loader.load()
            logger.info(f"Loaded {filename}: {len(documents)} pages")
            return documents
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []

