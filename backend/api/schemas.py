"""Request and Response schemas for FastAPI endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum



class RetrievalStrategyEnum(str, Enum):
    """Retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class PresetEnum(str, Enum):
    """Available presets."""
    SIMPLE = "simple"
    QA = "qa"
    RESEARCH = "research"




class QueryRequest(BaseModel):
    """Single query request."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    preset: Optional[str] = Field(None, description="Preset name: simple, qa, research")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="LLM temperature")
    top_k_retrieve: Optional[int] = Field(10, ge=1, le=100, description="Docs to retrieve")
    top_k_rerank: Optional[int] = Field(5, ge=1, le=50, description="Docs after reranking")
    enable_query_expansion: Optional[bool] = Field(True, description="Enable HyDE expansion")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is attention in transformers?",
                "preset": "research",
                "temperature": 0.5,
                "top_k_retrieve": 15,
                "top_k_rerank": 5
            }
        }


class SourceDocument(BaseModel):
    """Source document with metadata."""
    source_file: str = Field(..., description="PDF filename")
    page_number: int = Field(..., description="Page in source PDF")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content_preview: str = Field(..., description="First 200 chars of chunk")
    relevance_score: Optional[float] = Field(None, description="Reranker score")


class QueryMetadata(BaseModel):
    """Query execution metadata."""
    preset_used: Optional[str] = Field(None, description="Preset applied")
    retrieval_strategy: str = Field(..., description="dense, sparse, or hybrid")
    reranking_enabled: bool = Field(..., description="Whether reranking was applied")
    query_expansion: bool = Field(..., description="Whether query was expanded")
    num_queries_expanded: int = Field(..., description="Number of expanded queries")
    num_candidates_retrieved: int = Field(..., description="Initial retrieved docs")
    num_final_sources: int = Field(..., description="Final sources in response")
    timing_ms: float = Field(..., description="Total execution time in ms")


class QueryResponse(BaseModel):
    """Successful query response."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="LLM-generated answer")
    sources: List[SourceDocument] = Field(..., description="Source documents used")
    metadata: QueryMetadata = Field(..., description="Execution metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is attention?",
                "answer": "Attention is a mechanism that allows...",
                "sources": [
                    {
                        "source_file": "paper1.pdf",
                        "page_number": 5,
                        "chunk_id": "chunk_123",
                        "content_preview": "The attention mechanism works by...",
                        "relevance_score": 0.95
                    }
                ],
                "metadata": {
                    "preset_used": "research",
                    "retrieval_strategy": "hybrid",
                    "reranking_enabled": True,
                    "timing_ms": 1250.5
                }
            }
        }


class QueryErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    query: Optional[str] = Field(None, description="Original query (if available)")
    timing_ms: float = Field(..., description="Execution time before error")


class PresetInfo(BaseModel):
    """Information about a preset."""
    name: str = Field(..., description="Preset name")
    description: str = Field(..., description="What this preset is for")
    top_k_retrieve: int = Field(..., description="Initial retrievals")
    top_k_rerank: int = Field(..., description="Final results")
    temperature: float = Field(..., description="LLM temperature")


class PresetsResponse(BaseModel):
    """List of available presets."""
    presets: List[PresetInfo] = Field(..., description="Available presets")



class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    filename: str = Field(..., description="Uploaded filename")
    status: str = Field(..., description="File received status")
    message: str = Field(..., description="Details")


class DocumentInfo(BaseModel):
    """Information about a processed document."""
    filename: str = Field(..., description="PDF filename")
    pages: int = Field(..., description="Number of pages")
    chunks: int = Field(..., description="Number of chunks created")
    processed_at: str = Field(..., description="ISO timestamp")
    size_bytes: int = Field(..., description="File size in bytes")


class DocumentsResponse(BaseModel):
    """List of documents in the system."""
    documents: List[DocumentInfo] = Field(..., description="Processed documents")
    total_documents: int = Field(..., description="Total count")
    total_chunks: int = Field(..., description="Total chunks across all docs")


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""
    filename: str = Field(..., description="Deleted filename")
    chunks_removed: int = Field(..., description="Chunks that were deleted")
    message: str = Field(..., description="Success message")


class ProcessingRequest(BaseModel):
    """Request to process documents."""
    force_reprocess: bool = Field(False, description="Reprocess even unchanged files")


class ProcessingResponse(BaseModel):
    """Response from document processing."""
    new_documents: int = Field(..., description="Newly processed documents")
    updated_documents: int = Field(..., description="Re-processed documents")
    total_chunks: int = Field(..., description="Chunks generated")
    status: str = Field("success", description="Processing status")
    timing_ms: float = Field(..., description="Processing time")



class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status (OK/ERROR)")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="ISO timestamp")


class SystemStats(BaseModel):
    """System statistics."""
    total_documents: int = Field(..., description="Total PDFs processed")
    total_chunks: int = Field(..., description="Total chunks created")
    total_embeddings: int = Field(..., description="Total embeddings stored")
    chunk_store_size_mb: float = Field(..., description="JSONL file size")
    vector_store_size_mb: float = Field(..., description="Chroma DB size")
    avg_query_time_ms: Optional[float] = Field(None, description="Average query time")


class SystemStatusResponse(BaseModel):
    """System status response."""
    service: str = Field("PaperFlow RAG API", description="Service name")
    version: str = Field("1.0.0", description="API version")
    status: str = Field(..., description="Overall status")
    embedding_service: str = Field(..., description="Embedding model loaded")
    vector_db: str = Field(..., description="Vector DB status")
    chunk_store: str = Field(..., description="Chunk store status")
    stats: SystemStats = Field(..., description="System statistics")
    timestamp: str = Field(..., description="ISO timestamp")



class ConfigUpdate(BaseModel):
    """Configuration update request."""
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k_retrieve: Optional[int] = Field(None, ge=1, le=100)
    top_k_rerank: Optional[int] = Field(None, ge=1, le=50)
    retrieval_strategy: Optional[RetrievalStrategyEnum] = None
    enable_query_expansion: Optional[bool] = None
    reranking_enabled: Optional[bool] = None


class ConfigResponse(BaseModel):
    """Current configuration."""
    temperature: float
    top_k_retrieve: int
    top_k_rerank: int
    retrieval_strategy: str
    enable_query_expansion: bool
    reranking_enabled: bool



class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    field: str
    message: str


class BadRequestResponse(BaseModel):
    """Bad request response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ValidationErrorDetail]] = None


class ServerErrorResponse(BaseModel):
    """Server error response."""
    error: str = Field("Internal Server Error")
    message: str = Field(...)
    request_id: Optional[str] = None
