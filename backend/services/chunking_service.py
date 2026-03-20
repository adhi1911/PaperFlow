"""Text chunking using LangChain RecursiveCharacterTextSplitter."""
import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class ChunkingService:
    """Split Documents into chunks while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks. Adds chunk_index to metadata."""
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

