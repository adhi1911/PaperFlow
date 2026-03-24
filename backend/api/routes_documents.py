"""Document management endpoints: upload, list, delete, reprocess PDFs."""

import logging
import os
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, BackgroundTasks

from backend.api.schemas import (
    DocumentUploadResponse, DocumentsResponse, DocumentInfo,
    DocumentDeleteResponse, ProcessingRequest, ProcessingResponse
)
from backend.api.dependencies import get_paper_processor, get_chunk_store, get_service_container
from backend.utils.path_utils import get_relative_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a PDF document",
    description="Upload a PDF to be processed and indexed."
)
async def upload_document(
    file: UploadFile = File(...),
    processor = Depends(get_paper_processor),
):
    """
    Upload a PDF document.
    
    The document will be:
    1. Saved to data/raw_papers/
    2. Processed on next call to /process (or automatically if auto-processing is enabled)
    3. Chunked and embedded
    4. Indexed in vector store
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Save file
        papers_dir = processor.papers_dir
        papers_dir.mkdir(parents=True, exist_ok=True)
        file_path = papers_dir / file.filename
        
        contents = await file.read()
        
        # Check file size (max 50MB)
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large (max 50MB)"
            )
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"PDF uploaded: {file.filename} ({len(contents)} bytes)")
        
        return DocumentUploadResponse(
            filename=file.filename,
            status="uploaded",
            message=f"PDF uploaded successfully. Call /process to index."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get(
    "",
    response_model=DocumentsResponse,
    summary="List all documents",
    description="Get a list of all processed documents."
)
async def list_documents(chunk_store = Depends(get_chunk_store)):
    """
    List all processed documents with metadata.
    """
    try:
        all_chunks = chunk_store.load_all()
        
        # Group chunks by source file
        documents_map = {}
        
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            
            if source not in documents_map:
                documents_map[source] = {
                    "filename": source,
                    "chunks": 0,
                    "pages": set(),
                }
            
            documents_map[source]["chunks"] += 1
            page = chunk.metadata.get("page_number", 0)
            documents_map[source]["pages"].add(page)
        
        # Convert to response format
        container = get_service_container()
        documents = []
        for doc_data in documents_map.values():
            doc_info = DocumentInfo(
                filename=get_relative_path(doc_data["filename"], container.project_root),
                pages=max(doc_data["pages"]) + 1 if doc_data["pages"] else 0,
                chunks=doc_data["chunks"],
                processed_at="",  # Would come from metadata if stored
                size_bytes=0,  # Would need to stat the file
            )
            documents.append(doc_info)
        
        total_chunks = sum(d.chunks for d in documents)
        
        return DocumentsResponse(
            documents=documents,
            total_documents=len(documents),
            total_chunks=total_chunks
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list documents"
        )


@router.delete(
    "/{document_name}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Remove a document and all its chunks from the system."
)
async def delete_document(
    document_name: str,
    chunk_store = Depends(get_chunk_store),
):
    """
    Delete a document by filename.
    
    This will:
    1. Remove all chunks associated with this file
    2. Delete embeddings from vector store
    3. Update the registry
    """
    try:
        # Count chunks before deletion for response
        all_chunks = chunk_store.load_all()
        chunks_to_remove = [
            c for c in all_chunks
            if c.metadata.get("source") == document_name
        ]
        
        if not chunks_to_remove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_name}' not found"
            )
        
        # Remove from chunk store
        chunk_store.remove_by_source(document_name)
        
        logger.info(f"Document deleted: {document_name} ({len(chunks_to_remove)} chunks)")
        
        return DocumentDeleteResponse(
            filename=document_name,
            chunks_removed=len(chunks_to_remove),
            message=f"Document and {len(chunks_to_remove)} chunks deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}"
        )


@router.post(
    "/process",
    response_model=ProcessingResponse,
    summary="Process documents",
    description="Process uploaded PDFs: chunk, embed, and index."
)
async def process_documents(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
    processor = Depends(get_paper_processor),
):
    """
    Trigger document processing pipeline.
    
    This will:
    1. Detect new PDFs in data/raw_papers/
    2. Chunk each document (1000 char, 200 overlap)
    3. Generate embeddings (384-dim all-MiniLM-L6-v2)
    4. Store chunks in JSONL and vectors in Chroma
    5. Track changes with MD5 hashing
    
    Only changed/new documents are reprocessed (unless force_reprocess=true).
    """
    try:
        logger.info("Starting document processing...")
        
        import time
        start = time.time()
        
        # Run processing
        result = processor.process_all()
        
        elapsed_ms = (time.time() - start) * 1000
        
        logger.info(
            f"Processing complete: "
            f"{result.get('new_processed', 0)} new, "
            f"{result.get('updated_processed', 0)} updated, "
            f"{result.get('total_chunks', 0)} chunks in {elapsed_ms:.0f}ms"
        )
        
        return ProcessingResponse(
            new_documents=result.get("new_processed", 0),
            updated_documents=result.get("updated_processed", 0),
            total_chunks=result.get("total_chunks", 0),
            status="success",
            timing_ms=elapsed_ms
        )
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )

