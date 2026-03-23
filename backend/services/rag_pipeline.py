from dataclasses import dataclass , field 
from enum import Enum 
from typing import Dict, Any, List 
import logging
import time

from backend.services.embedding import EmbeddingService 
from backend.services.vector_store import VectorStore
from backend.services.chunk_store import ChunkStore
from backend.services.query_transformer import QueryTransformer
from backend.services.hybrid_retriever import HybridRetriever
from backend.services.reranker import CrossEncoderReRanker
from backend.services.generation import GroqGenerator

logger = logging.getLogger(__name__)

class RetrievalStrategy(str, Enum): 
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

@dataclass
class RAGConfig: 
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k_retrieve : int = 5 
    dense_weight: float = 0.6 
    sparse_weight: float = 0.4 
    enable_query_expansion: bool = False
    expansion_num_queries: int = 3

    reranking_enabled: bool = True 
    top_k_final : int = 5 
    rerank_threshold: float = 0.0

    temperature: float = 0.3
    system_message: str | None = None
    response_format: str = "narrative"
    include_reasoning: bool = False

    @classmethod 
    def from_preset(cls, preset_config: Dict[str, Any]) -> "RAGConfig": 

        retrieval = preset_config.get("retrieval", {})
        reranking = preset_config.get("reranking", {})
        generation = preset_config.get("generation", {})
        
        return cls(
            # Retrieval
            retrieval_strategy=RetrievalStrategy(retrieval.get("strategy", "hybrid")),
            top_k_retrieve=retrieval.get("top_k", 10),
            dense_weight=retrieval.get("dense_weight", 0.6),
            sparse_weight=retrieval.get("sparse_weight", 0.4),
            enable_query_expansion=retrieval.get("query_expansion", True),
            expansion_num_queries=retrieval.get("expansion_queries", 3),
            
            # Reranking
            reranking_enabled=reranking.get("enabled", True),
            top_k_final=reranking.get("top_k", 5),
            rerank_threshold=reranking.get("threshold", 0.0),
            
            # Generation
            temperature=generation.get("temperature", 0.3),
            system_message=generation.get("system_message"),
            response_format=generation.get("response_format", "narrative"),
            include_reasoning=generation.get("include_reasoning", False),
        )


#### RAG PIPEPLINE 
class RAGPipeline: 
    """Master orchestrator class """

    def __init__(self,
                chunk_store_path: str,
                vector_store_path: str, 
                preset_registry = None, 
                ):
        
        # initializing all services 
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(persist_directory=vector_store_path)
        self.chunk_store = ChunkStore(chunk_store_path)
        self.query_transformer = QueryTransformer()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            documents=self.chunk_store.load_all()
        )
        self.reranker = CrossEncoderReRanker()
        self.generator = GroqGenerator()

        self.preset_registry = preset_registry 

    def query(self, 
                user_query: str, 
                preset: str | None = None, 
                config: RAGConfig | None = None,
                **kwargs
            ) -> Dict[str, Any]:

        start_time = time.time()

        if config is None: 
            if preset and self.preset_registry: 
                preset_dict = self.preset_registry.get(preset)
                config = RAGConfig.from_preset(preset_dict)

            else: 
                config = RAGConfig()

        for key, value in kwargs.items(): 
            if hasattr(config, key):
                setattr(config, key, value)

        try:
            # expanding query 
            queries = self._step_expand_query(user_query, config) 

            # retrieving documents
            candidates = self._step_retrieve(queries, config)

            # reranking 
            reranked = self._step_rerank(user_query, candidates, config)

            # final generation 
            answer = self._step_generate(user_query, reranked, config)


            result = {
                "query": user_query,
                "answer": answer.get("answer", ""),
                "sources": answer.get("sources", []),
                "num_sources": len(answer.get("sources", [])),
                "metadata": {
                    "preset_used": preset,
                    "retrieval_strategy": config.retrieval_strategy.value,
                    "reranking_enabled": config.reranking_enabled,
                    "query_expansion": config.enable_query_expansion,
                    "num_queries_expanded": len(queries),
                    "num_candidates_retrieved": len(candidates),
                },
                "timing_ms": (time.time() - start_time) * 1000,
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                "error": str(e),
                "query": user_query,
                "timing_ms": (time.time() - start_time) * 1000,
            }


    def _step_expand_query(self, query:str, config:RAGConfig) -> List[str]:
        """query transformer to expand query"""
        if not config.enable_query_expansion:
            return [query]
        return self.query_transformer.transform(
            query, 
            num_multi_queries=config.expansion_num_queries,
            use_hyde=True,
            use_multi=True
        )
    

    def _step_retrieve(self, queries: List[str], config: RAGConfig) -> List[Dict]:
        results = {}
        
        for query in queries:
            embedding = self.embedding_service.embed_texts([query])[0]
            
            if config.retrieval_strategy == RetrievalStrategy.HYBRID:
                retrieved = self.hybrid_retriever.search(
                    embedding, query,
                    top_k=config.top_k_retrieve
                )
            elif config.retrieval_strategy == RetrievalStrategy.DENSE:
                retrieved = self.vector_store.search(
                    embedding,
                    top_k=config.top_k_retrieve
                )
            
            # deduplicating using chunk id
            for doc in retrieved:
                cid = doc["chunk_id"]
                if cid not in results:
                    results[cid] = doc
    
        return list(results.values())
    

    def _step_rerank(self, query: str, candidates: List[Dict], 
                     config: RAGConfig) -> List[Dict]:
        if not config.reranking_enabled or not candidates:
            return candidates[:config.top_k_final]
        
        return self.reranker.rerank(
            query, candidates,
            top_k=config.top_k_final,
            score_threshold=config.rerank_threshold
        )
    
    def _step_generate(self, query: str, documents: List[Dict], 
                       config: RAGConfig) -> Dict:
        from backend.services.generation import GenerationConfig, ResponseFormat
        
        gen_config = GenerationConfig(
            system_message=config.system_message,
            response_format=ResponseFormat(config.response_format),
            temperature=config.temperature,
            include_reasoning=config.include_reasoning,
        )
        
        return self.generator.generate_with_metadata(query, documents, gen_config)

        