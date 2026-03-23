"""Hybrid retriever combining dense + sparese retrieval"""

import logging 
from typing import List, Dict, Any 

import numpy as np 
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class HybridRetriever: 

    def __init__(self, 
                 vector_store,
                 documents: List[Document],
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4
            ):
        
        self.vector_store = vector_store
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.documents = documents
        self.metadata_to_doc = {
            doc.metadata.get("source") + "_" + str(doc.metadata.get("page"))
            + "_" + str(doc.metadata.get("chunk_index")): doc
            for doc in documents
        }

        corpus = [
            doc.page_content.lower().split() for doc in documents
        ]

        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index ready with {len(documents)} documents")

    def _dense_search(self,
                      query_embedding: np.ndarray, 
                      top_k: int     
            ) -> List[Dict[str, Any]]:
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
    
    def _bm25_search(self, 
                    query:str, 
                    top_k:int) -> List[Dict[str,Any]]: 
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [] 
        for idx in top_indices: 
            if idx < len(self.documents): 
                doc = self.documents[idx]
                results.append({
                    "chunk_id": f"bm25_{idx}",
                    "content" : doc.page_content,
                    "metadata": doc.metadata,
                    "bm25_score": float(scores[idx])
                })

        return results
    

    # rrf 
    def rrf(self,
            dense_results: List[Dict[str, Any]],
            sparse_results: List[Dict[str, Any]],
            top_k: int = 10 
        ) -> List[Dict[str, Any]]:

        """Reciprocal Rank Fusion of dense and sparse results."""
        scores = {}

        for rank, result in enumerate(dense_results):
            chunk_id = result["chunk_id"]
            score = 1/(60 + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + self.dense_weight * score

        for rank, result in enumerate(sparse_results):
            chunk_id = result["chunk_id"]
            score = 1/(60 + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + self.sparse_weight * score

        combined = {}
        for result in dense_results + sparse_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in combined:
                combined[chunk_id] = result

        ranked = sorted(combined.items(), key=lambda x: scores.get(x[0], 0), reverse=True)

        final_results = []
        for chunk_id , result in ranked[:top_k]: 
            result["combined_score"] = scores.get(chunk_id, 0)
            final_results.append(result)

        return final_results 

    
    def search(self,
                query_embedding: np.ndarray, 
                query:str, 
                top_k:int = 10
            ) -> List[Dict[str, Any]]:

            dense_results = self._dense_search(query_embedding, top_k=top_k*5)
            sparse_results = self._bm25_search(query, top_k=top_k*5)

            results = self.rrf(dense_results, sparse_results, top_k=top_k)

            return results
        
