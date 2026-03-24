"""FastAPI main application with all routes and middleware."""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# Import routes
from backend.api.routes_query import router as query_router
from backend.api.routes_documents import router as documents_router
from backend.api.routes_system import router as system_router
from backend.api.dependencies import get_service_container

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ LIFECYCLE EVENTS ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    logger.info(" PaperFlow RAG API starting...")
    try:
        # Initialize core services (RAGPipeline is lazy-loaded)
        container = get_service_container()
        logger.info(f"[OK] Core services initialized")
        logger.info(f"  - Chunk store: {container.get_relative_path(container.chunk_store_path)}")
        logger.info(f"  - Vector store: {container.get_relative_path(container.vector_store_path)}")
        logger.info(f"  - Papers dir: {container.get_relative_path(container.papers_dir)}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize core services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info(" PaperFlow RAG API shutting down...")


# ============ CREATE APP ============

app = FastAPI(
    title="PaperFlow RAG API",
    description="Retrieval-Augmented Generation for research papers",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============ MIDDLEWARE ============

# CORS middleware - allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    
    # logger.info(f"{request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # loggin response
        logger.info(
            f"  → {response.status_code} ({duration*1000:.1f}ms)"
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"  → ERROR ({duration*1000:.1f}ms): {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# routes 

# Include routers
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(system_router)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs."""
    return {
        "service": "PaperFlow RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/system/health",
        "status": "/api/system/status",
    }


# error handling

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Endpoint {request.url.path} not found",
            "docs": "/docs"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )



if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting server...")
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
