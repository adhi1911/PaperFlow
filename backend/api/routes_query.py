"""Query endpoints: execute RAG queries and manage presets."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from pathlib import Path

from backend.api.schemas import (
    QueryRequest, QueryResponse, QueryErrorResponse,
    PresetsResponse, PresetInfo
)
from backend.api.dependencies import get_rag_pipeline, get_preset_registry, get_service_container
from backend.utils.path_utils import get_relative_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": QueryErrorResponse, "description": "Invalid query"},
        500: {"model": QueryErrorResponse, "description": "Query processing error"},
    },
    summary="Execute a RAG query",
    description="Submit a query and get an answer with source citations."
)
async def execute_query(
    request: QueryRequest,
    rag_pipeline = Depends(get_rag_pipeline),
):
    """
    Execute a single query against the document corpus.
    
    Returns answer with source citations and execution metadata.
    """
    try:
        logger.info(f"Query received: {request.query[:50]}...")
        
        # Build kwargs from request
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_k_retrieve is not None:
            kwargs["top_k_retrieve"] = request.top_k_retrieve
        if request.top_k_rerank is not None:
            kwargs["top_k_final"] = request.top_k_rerank
        if request.enable_query_expansion is not None:
            kwargs["enable_query_expansion"] = request.enable_query_expansion
        
        # Execute query
        result = rag_pipeline.query(
            user_query=request.query,
            preset=request.preset or "qa",
            **kwargs
        )
        
        # Check for errors in result
        if "error" in result:
            logger.error(f"Pipeline error: {result['error']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Transform result to response format
        container = get_service_container()
        sources = [
            {
                "source_file": get_relative_path(src.get("source", "unknown"), container.project_root),
                "page_number": src.get("page", 0),
                "chunk_id": src.get("chunk_id", ""),
                "content_preview": src.get("content", "")[:200],
                "relevance_score": src.get("score"),
            }
            for src in result.get("sources", [])
        ]
        
        metadata = result.get("metadata", {})
        
        response = QueryResponse(
            query=result["query"],
            answer=result.get("answer", ""),
            sources=sources,
            metadata={
                "preset_used": metadata.get("preset_used"),
                "retrieval_strategy": metadata.get("retrieval_strategy", "hybrid"),
                "reranking_enabled": metadata.get("reranking_enabled", True),
                "query_expansion": metadata.get("query_expansion", False),
                "num_queries_expanded": metadata.get("num_queries_expanded", 1),
                "num_candidates_retrieved": metadata.get("num_candidates_retrieved", 0),
                "num_final_sources": len(sources),
                "timing_ms": result.get("timing_ms", 0),
            }
        )
        
        logger.info(f"Query executed successfully in {result.get('timing_ms', 0):.0f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get(
    "/presets",
    response_model=PresetsResponse,
    summary="List available presets",
    description="Get information about all available query presets."
)
async def get_presets(preset_registry = Depends(get_preset_registry)):
    """
    List all available query presets with their configurations.
    
    Presets are pre-configured settings for different query types:
    - simple: Fast basic retrieval
    - qa: Balanced retrieval (default)
    - research: Comprehensive deep synthesis
    """
    try:
        preset_names = preset_registry.list_presets()
        
        presets = []
        descriptions = {
            "simple": "Fast basic retrieval with minimal reranking",
            "qa": "Balanced retrieval suitable for Q&A (default)",
            "research": "Comprehensive synthesis with aggressive reranking"
        }
        
        defaults = {
            "simple": {"top_k_retrieve": 5, "top_k_rerank": 3, "temperature": 0.5},
            "qa": {"top_k_retrieve": 10, "top_k_rerank": 5, "temperature": 0.6},
            "research": {"top_k_retrieve": 25, "top_k_rerank": 10, "temperature": 0.7},
        }
        
        for name in preset_names:
            config = defaults.get(name, {})
            presets.append(PresetInfo(
                name=name,
                description=descriptions.get(name, "Custom preset"),
                top_k_retrieve=config.get("top_k_retrieve", 10),
                top_k_rerank=config.get("top_k_rerank", 5),
                temperature=config.get("temperature", 0.6),
            ))
        
        return PresetsResponse(presets=presets)
        
    except Exception as e:
        logger.error(f"Error fetching presets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve presets"
        )


@router.get(
    "/presets/{preset_name}",
    summary="Get preset details",
    description="Get detailed configuration for a specific preset."
)
async def get_preset_details(
    preset_name: str,
    preset_registry = Depends(get_preset_registry)
):
    """
    Get detailed configuration for a specific preset.
    """
    try:
        if preset_name not in preset_registry.list_presets():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset '{preset_name}' not found"
            )
        
        config = preset_registry.get(preset_name)
        
        return {
            "name": preset_name,
            "config": config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching preset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preset"
        )
