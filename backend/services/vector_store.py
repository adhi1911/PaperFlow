"""Persist and query text chunks with metadata in a vector store."""

import logging 
from pathlib import Path 
from typing import List , Dict, Any

import numpy as np 
import chromadb 

logger = logging.getLogger(__name__)

class VectorStore: 

    def __init__(self, 
                 collection_name: str = "paper_chunks",
                 persist_directory: Path = Path("./vector_store"),
    ):
        self.collection_name = collection_name 
        self.persist_directory = Path(persist_directory)
        self.client = None 
        self.collection = None 
        self._initialize() 

    def _initialize(self): 
        try: 
            self.persist_directory.mkdir(parents = True, exist_ok = True)
            self.client = chromadb.PersistentClient(str(self.persist_directory))

            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {"description": "Chunks of academic papers with metadata"}
            )

            logger.info(f"Initialized vector store at {self.persist_directory} with collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def add_embeddings(self, 
                        embedding_results: List[Dict[str,Any]],
                        auto_delete_source: bool = True
        ) -> int:
        """Add embeddings from EmbeddingService"""

        if not embedding_results:
            return 0 
        
        if auto_delete_source: 
            sources = set(
                r["metadata"].get("source")
                for r in embedding_results
                if "source" in r["metadata"]
            )
            for source in sources: 
                deleted = self.delete_by_source(source)
                if deleted> 0: 
                    logger.info(f"Auto-deleted {deleted} existing vectors from source: {source}")

        try:
            chunkd_ids = [r['chunk_id'] for r in embedding_results]
            embeddings = [r['embedding'] for r in embedding_results]
            metadatas = [r['metadata'] for r in embedding_results]
            texts = [r['content'] for r in embedding_results]

            self.collection.add(
                ids = chunkd_ids,
                embeddings = embeddings,
                metadatas = metadatas,
                documents = texts
            )

            logger.info(f"Added {len(embedding_results)} embeddings to vector store")
            return len(embedding_results)
        
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            raise

        
    def search(self,
                query_embedding: np.ndarray, 
                top_k : int = 5,
                score_threshold: float = 0.0,
        ) -> List[Dict[str, Any]]:
        """Search for similar chunks given a query embedding."""

        try: 
            results = self.collection.query(
                query_embeddings = [query_embedding.tolist()],
                n_results = top_k,
            )

            retrieved = []
            if results["documents"] and results["documents"][0]:
                for i, (doc_id, text, metadata, score) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    similarity_score = 1 - score 

                    if similarity_score >= score_threshold:
                        retrieved.append({
                            "chunk_id": doc_id,
                            "content": text,
                            "metadata": metadata,
                            "similarity_score": similarity_score
                        })

            logger.debug(f"Search retrieved {len(retrieved)} results (threshold: {score_threshold})")
            return retrieved
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def delete_by_source(self, source: str) -> int:
        """Delete all vectors from a source (paper)."""
        try:
            all_data = self.collection.get()
            ids_to_delete = [
                doc_id
                for doc_id, metadata in zip(all_data["ids"], all_data["metadatas"])
                if metadata.get("source") == source
            ]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} vectors from {source}")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Error deleting by source: {e}")
            raise        

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            all_data = self.collection.get()

            sources = set()
            for metadata in all_data.get("metadatas", []):
                if "source" in metadata:
                    sources.add(metadata["source"])

            return {
                "total_vectors": count,
                "total_sources": len(sources),
                "sources": sorted(list(sources)),
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"VectorStore(vectors={stats.get('total_vectors', 0)}, sources={stats.get('total_sources', 0)})"