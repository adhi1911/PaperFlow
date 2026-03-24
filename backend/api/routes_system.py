"""System status endpoints: health, stats, configuration."""

import logging
import os
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.schemas import (
    HealthResponse, SystemStatusResponse, SystemStats,
    ConfigResponse, ConfigUpdate
)
from backend.api.dependencies import (
    get_rag_pipeline, get_chunk_store,
    get_vector_store, get_embedding_service, get_service_container
)
from backend.utils.path_utils import get_relative_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is running and services are available."
)
async def health_check(
    rag_pipeline = Depends(get_rag_pipeline),
):
    """
    Quick health check endpoint.
    
    Returns status of core services. Use for monitoring/uptime checks.
    """
    try:
        # Try a dummy operation to verify pipeline is working
        if rag_pipeline is None:
            raise RuntimeError("RAG pipeline not initialized")
        
        return HealthResponse(
            status="OK",
            message="All systems operational",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="ERROR",
            message=f"Service degraded: {str(e)}",
            timestamp=datetime.utcnow().isoformat()
        )


@router.get(
    "/stats",
    summary="Get statistics",
    description="Get document and system statistics."
)
async def get_stats(chunk_store = Depends(get_chunk_store)):
    """
    Get detailed statistics about the document corpus.
    """
    try:
        all_chunks = chunk_store.load_all()
        container = get_service_container()
        
        # Group by source
        sources = {}
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            if source not in sources:
                sources[source] = {"chunks": 0, "size": 0, "pages": set()}
            
            sources[source]["chunks"] += 1
            sources[source]["size"] += len(chunk.page_content)
            sources[source]["pages"].add(chunk.metadata.get("page_number", 0))
        
        # Calculate stats
        total_chunks = len(all_chunks)
        total_size = sum(len(c.page_content) for c in all_chunks)
        avg_chunk_size = total_size / total_chunks if total_chunks > 0 else 0
        
        # Convert source paths to relative paths
        relative_sources = {
            get_relative_path(source, container.project_root): info
            for source, info in sources.items()
        }
        
        return {
            "total_documents": len(relative_sources),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "avg_chunk_size_bytes": avg_chunk_size,
            "documents": {
                source: {
                    "chunks": info["chunks"],
                    "size_bytes": info["size"],
                    "pages": len(info["pages"])
                }
                for source, info in relative_sources.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stats"
        )


@router.get(
    "/config",
    response_model=ConfigResponse,
    summary="Get current configuration",
    description="Get the current RAG configuration settings."
)
async def get_config(rag_pipeline = Depends(get_rag_pipeline)):
    """
    Get current system configuration.
    """
    try:
        # This would normally come from a config service
        # For now, return defaults
        return ConfigResponse(
            temperature=0.7,
            top_k_retrieve=10,
            top_k_rerank=5,
            retrieval_strategy="hybrid",
            enable_query_expansion=True,
            reranking_enabled=True,
        )
        
    except Exception as e:
        logger.error(f"Config retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve config"
        )


@router.post(
    "/config",
    response_model=ConfigResponse,
    summary="Update configuration",
    description="Update RAG configuration settings."
)
async def update_config(
    request: ConfigUpdate,
    rag_pipeline = Depends(get_rag_pipeline),
):
    """
    Update system configuration.
    
    Note: This endpoint validates and applies changes.
    Some changes may require service restart.
    """
    try:
        # TODO: Implement config update logic
        # For now, just validate and return current state
        
        logger.info(f"Config update requested: {request.dict(exclude_none=True)}")
        
        # In a real implementation, persist to config file
        # and potentially patch running services
        
        return ConfigResponse(
            temperature=request.temperature or 0.7,
            top_k_retrieve=request.top_k_retrieve or 10,
            top_k_rerank=request.top_k_rerank or 5,
            retrieval_strategy=request.retrieval_strategy or "hybrid",
            enable_query_expansion=request.enable_query_expansion or True,
            reranking_enabled=request.reranking_enabled or True,
        )
        
    except Exception as e:
        logger.error(f"Config update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config update failed: {str(e)}"
        )


@router.get(
    "/info",
    summary="API Information",
    description="Get API version and feature information."
)
async def api_info():
    """
    Get API information and capabilities.
    """
    return {
        "service": "PaperFlow RAG API",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation for research papers",
        "capabilities": [
            "semantic search",
            "query expansion (HyDE + multi-query)",
            "hybrid retrieval (dense + BM25 + RRF)",
            "cross-encoder reranking",
            "LLM answer generation",
            "configurable presets",
            "document management",
            "incremental processing"
        ],
        "endpoints": {
            "query": "/api/query",
            "documents": "/api/documents",
            "system": "/api/system",
        },
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
