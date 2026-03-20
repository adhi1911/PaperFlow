"""Generate embeddings for document chunks."""
import logging
from typing import List 

import numpy as np 
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService: 
    """Generate dense embeddings using Sentence-Transformers"""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2"
                ):
        self.model_name = model_name
        self.model = None 
        self.embedding_dim = None
        self._load_model()

    def _load_model(self):
        try: 
            logger.info(f"Loading embedding model: {self.model_name}")
            # Suppress HuggingFace Hub verbose logs during model download
            logging.getLogger("httpx").setLevel(logging.WARNING)
            logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
            
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def embed_texts(self,
                    texts: List[str], 
                    batch_size: int = 32
                    ) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts: 
            return np.array([])
        
        try: 
            embeddings = self.model.encode(texts, 
                                            batch_size=batch_size,
                                            show_progress_bar=True,
                                            convert_to_numpy=True)
            
            return embeddings 
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_chunks(self, 
                    chunks: List[Document],
                    batch_size: int = 32
                    ) -> np.ndarray:
        """Generate embeddings for a list of Document chunks."""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        results = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"chunk_{chunk.metadata.get('source','unknown')}_{i}"

            results.append({
                "chunk_id": chunk_id,
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata
            })

        return results 
    
    def get_embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        return self.embedding_dim
    
    def __repr__(self): 
        return f"EmbeddingService(model_name='{self.model_name}', dim={self.embedding_dim})"