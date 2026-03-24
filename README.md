<div align="center">

# 🚀 PaperFlow

**Production-ready RAG system that turns PDFs into intelligent Q&A**

[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen?style=flat-square)](#)
[![Python](https://img.shields.io/badge/python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](#license)
[![Groq](https://img.shields.io/badge/LLM-Groq%2FMixtral-purple?style=flat-square)](https://console.groq.com/)

**12 REST Endpoints** · **Semantic Search** · **Hybrid Retrieval** · **Production Security**

</div>

---

## ✨ Features

<table>
<tr>
<td>

### 🎯 Smart Document Processing
- Automatic PDF loading → chunking → embedding
- MD5 incremental processing (skip unchanged files)
- 384-dim sentence embeddings (all-MiniLM-L6-v2)
- LangChain metadata preservation end-to-end

</td>
<td>

### 🔍 Advanced Retrieval
- Hybrid search (dense + BM25 + RRF)
- Cross-encoder reranking for relevance
- Query expansion (HyDE)
- Configurable via YAML presets

</td>
</tr>
<tr>
<td>

### 🌐 Production REST API
- 12 endpoints for queries, documents, system
- Pydantic validation + type safety
- Swagger/OpenAPI docs auto-generated
- CORS + dependency injection

</td>
<td>

### 🔒 Security & Deployment
- Relative paths in logs (no information leakage)
- Environment variable management (.env)
- Docker-ready with health checks
- Graceful error handling + logging

</td>
</tr>
</table>

---

## 📊 Status

| Component | Status | Details |
|-----------|--------|---------|
| **Document Pipeline** | ✅ Complete | 18 papers → 1,387 chunks → embedded |
| **Retrieval & Generation** | ✅ Complete | Hybrid search + reranking + LLM |
| **REST API** | ✅ Complete | 12 production endpoints |
| **Security** | ✅ Complete | Relative paths, env vars, error handling |
| **Deployment** | ✅ Ready | Docker + production configuration |

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/paperflow.git
cd paperflow

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configuration

Create `.env` file:
```env
GROQ_API_KEY=your_api_key_here
```

### 3️⃣ Process Documents

```bash
# Copy PDFs to data/raw_papers/
cp /path/to/papers/*.pdf data/raw_papers/

# Process: load → chunk → embed → index
python -m backend.init_rag
```

Output:
```
================-================ 
PaperFlow RAG Initialization
================+================
Step 1: Processing papers...
[OK] 18 new papers processed
[OK] 1,387 total chunks

Step 2: Generating embeddings...
[OK] Generated 1,387 embeddings

Step 3: Storing embeddings in vector database...
[OK] Added 1,387 embeddings to vector store

Initialization Complete!
```

### 4️⃣ Start API Server

```bash
uvicorn backend.api.main:app --reload
```

🌐 Open http://localhost:8000/docs for interactive API testing

### 5️⃣ Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is attention in transformers?",
    "preset": "research"
  }'
```

---

## 📚 API Endpoints (12 Total)

### 🔎 Query Endpoints

```
POST   /api/query                       Execute RAG query with answer + sources
GET    /api/query/presets               List available presets (simple, qa, research)
GET    /api/query/presets/{name}        Get preset configuration
```

### 📄 Document Management

```
POST   /api/documents/upload            Upload PDF files
GET    /api/documents                   List all processed documents
DELETE /api/documents/{name}            Delete document + remove chunks
POST   /api/documents/process           Process all PDFs (chunk + embed + index)
```

### ⚙️ System Status

```
GET    /api/system/health               Health check
GET    /api/system/stats                Document & chunk statistics
GET    /api/system/config               Get system configuration
POST   /api/system/config               Update configuration
GET    /api/system/info                 API capabilities & version
```

📖 **[Full API Documentation](./API.md)** with examples

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PaperFlow RAG System                        │
└─────────────────────────────────────────────────────────────────┘

📥 INPUT
  └─ PDFs (data/raw_papers/)

📦 PROCESSING PIPELINE
  ├─ DocumentLoader      [Extract text + metadata]
  ├─ ChunkingService     [Split 1000 chars, 200 overlap]
  ├─ ChunkStore          [Persist to JSONL]
  └─ PaperRegistry       [Track via MD5 hash]

🧠 EMBEDDING & INDEXING
  ├─ EmbeddingService    [Generate 384-dim vectors]
  └─ VectorStore         [Index in Chroma DB]

🔍 RETRIEVAL & GENERATION
  ├─ HybridRetriever     [Dense + BM25 + RRF fusion]
  ├─ CrossEncoderReranker [Rerank by relevance]
  └─ GroqGenerator       [Generate answer with Groq LLM]

📤 OUTPUT
  └─ Answer + Citations + Metadata
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Framework** | FastAPI | Type-safe, automatic OpenAPI docs |
| **Embeddings** | Sentence-Transformers | 384-dim, all-MiniLM-L6-v2 |
| **Vector DB** | Chroma | Local, persistent, ANN-ready |
| **Sparse Search** | BM25 | Keyword matching + rank_bm25 |
| **Hybrid Fusion** | RRF | Reciprocal Rank Fusion |
| **Reranking** | Cross-Encoder | ms-marco-MiniLM-L6-v2 |
| **LLM** | Groq (Mixtral) | Fast, $0.27/1M tokens |
| **Config** | YAML + Pydantic | External config + type-safety |
| **Storage** | JSONL | Queryable chunks |
| **Container** | Docker | Deploy anywhere |

---

## 📖 Usage Examples

### Python (Notebook)

>> One-Time Load
RAGPipeline initializes all services once, caches them. Query calls reuse cached services (fast + efficient).

>> Metadata Preservation
Document objects carry source file, page number, char position through entire pipeline. Citations work correctly.

>> Incremental Processing
Only process new/changed PDFs (MD5 hashing). Don't re-embed the entire corpus.

>> Modular Services
Each service has ONE responsibility. Swap implementations without affecting others.

## Data Flow

```
Raw Papers (PDF)
    |
    +-- DocumentLoader: Extract text + metadata
    |
    +-- ChunkingService: Split 1000 char, 200 overlap
    |
    +-- ChunkStore: Persist to JSONL
    |
    +-- PaperRegistry: Track via MD5 hash
    |
    +-- EmbeddingService: Generate 384-dim vectors (all-MiniLM-L6-v2)
    |
    +-- VectorStore: Index in Chroma (ANN search)
    |
Query (text)
    |
    +-- HybridRetriever: Dense search + BM25 + RRF
    |
    +-- CrossEncoderReranker: Rerank top-k by relevance
    |
    +-- GroqGenerator: Invoke LLM with context + system prompt
    |
    +-- [Answer + citations]
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Documents | LangChain | Metadata preservation, ecosystem |
| Embeddings | Sentence-Transformers | 384-dim, all-MiniLM-L6-v2 model |
| Vector DB | Chroma | Local, persistent, ANN-ready |
| Dense Retrieval | Cosine Similarity | Speed + relevance |
| Sparse Retrieval | BM25 | Keyword matching, rank_bm25 |
| Hybrid Fusion | RRF | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder | ms-marco-MiniLM-L6-v2 |
| LLM | Groq (Mixtral) | $0.27/1M tokens, fast inference |
| Config | YAML + Pydantic | External config, type-safe |
| Storage | JSONL | Simple, queryable chunks |

## Current Status

### ✅ Phase 1: Document Pipeline - COMPLETE
- 18 research papers loaded and processed
- 1,387 chunks created and persisted (JSONL format)
- All chunks embedded (384-dim, all-MiniLM-L6-v2)
- MD5 incremental processing tested and working
- Automatic cleanup of old embeddings

### ✅ Phase 2: Retrieval & Generation - COMPLETE
- Hybrid retrieval (dense + BM25 + RRF) implemented
- Cross-encoder reranking tested
- Query expansion (HyDE) working
- LLM generation with Groq (Mixtral) integrated
- Full RAG pipeline tested in examples.ipynb
- YAML preset system for configuration

### ✅ Phase 3: REST API - COMPLETE
- 12 production endpoints implemented
- FastAPI with proper error handling
- Request/response validation (Pydantic)
- Dependency injection pattern
- Middleware for logging and CORS
- Full API documentation (Swagger/OpenAPI)

### ✅ Phase 4: Security & Deployment - COMPLETE
- Environment variable loading (.env support)
- Relative paths in all logs and responses (no absolute paths exposed)
- Dockerfile and containerization ready
- Docker health checks configured
- Production-ready error handling
- Lazy loading of RAGPipeline (graceful initialization)

### 🚀 Deployment Ready
All components are production-ready and tested. Deploy to cloud with confidence.

## Presets

>> simple
Fast, basic retrieval. Fewer documents, minimal reranking, speed-optimized.

>> qa
Balanced retrieval. Standard document count, normal reranking, good quality.

>> research
Comprehensive synthesis. More documents, aggressive reranking, higher token usage.

## Philosophy

RAG done right means:
- Clean modular code over monoliths
- Incremental processing over full reprocessing
- YAML config over hardcoded values
- Type safety via Pydantic
- Metadata preservation end-to-end
- Test-friendly service boundaries
- Relative paths in logs (security)
- Environment variables for configuration
- Graceful error handling and logging
- Production-ready deployment

## Testing & Validation

### Python Examples (Notebook)

See `examples.ipynb` for complete usage examples:
- Document loading and chunking
- Embedding generation
- Vector store operations
- Hybrid retrieval
- Query expansion
- Full RAG pipeline queries

Run the notebook:
```bash
jupyter notebook examples.ipynb
```

### API Testing

Swagger UI for interactive testing:
```
http://localhost:8000/docs
```


### Integration Testing

All services are independently testable:
```python
from backend.services.paper_processor import PaperProcessor
from backend.services.rag_pipeline import RAGPipeline

# Test document processing
processor = PaperProcessor()
result = processor.process_all()
assert result['total_chunks'] > 0

# Test RAG queries
rag = RAGPipeline(...)
response = rag.query("What is AI?", preset="simple")
assert 'answer' in response
assert 'sources' in response
```

## Production Deployment

### Environment Setup

Required environment variables:
```
GROQ_API_KEY=your_api_key
```

Optional:
```
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Logging

All logs use relative paths (security):
```
[INFO] Initialized vector store at data/vector_store
[INFO] Loaded 42 documents from data/raw_papers
```

No absolute paths are exposed in logs or API responses.

### Performance

- **Initial setup:** ~30-60 seconds (18 papers → 1,387 chunks → embedded)
- **Query response:** <1 second (with reranking)
- **Memory footprint:** ~500MB (vector store + models in RAM)

### Scaling

To scale beyond 1,000 chunks:
1. Swap Chroma for Milvus/Weaviate/Qdrant (see `backend/services/vector_store.py`)
2. Add batch processing for embeddings
3. Consider distributed document processing
4. Use GPU for embeddings (CUDA/ROCm)

---

## API Reference

All code examples organized by use case.

### Basic Setup

```python
from backend.services.document_loader import DocumentLoader
from backend.services.chunking_service import ChunkingService
from backend.services.chunk_store import ChunkStore
from backend.services.paper_registry import PaperRegistry
from backend.services.paper_processor import PaperProcessor
from backend.services.embedding import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.rag_pipeline import RAGPipeline
from backend.services.preset_registry import PresetRegistry

# One call processes everything
processor = PaperProcessor()
result = processor.process_all()
print(f"Processed: {result['new_processed']} new, {result['updated_processed']} updated")
```

### Document Loading

```python
loader = DocumentLoader()
# Load all PDFs from data/raw_papers/
docs = loader.load_all_pdfs()

# Or single file
docs = loader.load_specific_pdf("my_paper.pdf")
```

### Chunking & Storage

```python
chunker = ChunkingService()
chunks = chunker.chunk_documents(docs)

store = ChunkStore("data/chunks.jsonl")
store.append_chunks(chunks)

# Remove by source file
store.remove_by_source("old_paper.pdf")

# Query storage
all_chunks = store.load_all()
stats = store.get_stats()
```

### Paper Registry

```python
reg = PaperRegistry()

# Get new/changed papers
new_papers = reg.get_new_papers("data/raw_papers/")

# Get processing status
status = reg.get_status()
```

### Embeddings

```python
embedding = EmbeddingService(model_name="all-MiniLM-L6-v2")

# Embed standalone texts
embeddings = embedding.embed_texts(["hello world", "how are you"])

# Embed Document chunks (preserves metadata)
embedded_chunks = embedding.embed_chunks(chunks)
```

### Vector Store

```python
vs = VectorStore(persist_directory="data/vector_store")

# Add embeddings
vs.add_embeddings(embedded_chunks)

# Search
results = vs.search(query_embedding, top_k=5)
# Returns: [{id, metadata, distance}, ...]

# Cleanup old embeddings
vs.delete_by_source("old_paper.pdf")
```

### Hybrid Retrieval

```python
from backend.services.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    chunked_documents=chunks,
    vector_store=vs
)

# Returns top_k results from dense + BM25 fusion
results = retriever.retrieve("machine learning", top_k=10)
```

### Query Transformation

```python
from backend.services.query_transformer import QueryTransformer

transformer = QueryTransformer(llm=groq_llm)

# HyDE + multi-query expansion
queries = transformer.transform("What is AI?")
# Returns: ["What is artificial intelligence?", "Define AI", ...]
```

### Reranking

```python
from backend.services.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()

# Rerank retrieved documents by relevance
reranked = reranker.rerank(
    query="neural networks",
    documents=retrieval_results,
    top_k=3
)
```

### Generation

```python
from backend.services.generation import GroqGenerator

generator = GroqGenerator(
    model_name="mixtral-8x7b-32768",
    system_prompt="You are a research assistant. Cite sources."
)

answer = generator.generate(
    query="Explain backpropagation",
    context=retrieved_docs,
    citation_style="inline"  # or "footnote", "numbered"
)
```

### Full RAG Pipeline

```python
from backend.services.rag_pipeline import RAGPipeline
from backend.services.preset_registry import PresetRegistry

rag = RAGPipeline(
    chunk_store_path="data/chunks.jsonl",
    vector_store_path="data/vector_store",
    preset_registry=PresetRegistry()
)

# Use preset
result = rag.query(
    query="What are the applications of deep learning?",
    preset="research",  # simple, qa, research
    temperature=0.7
)

print(result["answer"])
print(result["source_documents"])  # Metadata for citations

# Custom on-the-fly config
result = rag.query(
    query="Explain attention mechanisms",
    top_k_retrieve=15,  # Get more initial results
    top_k_rerank=5,     # Keep top 5 after reranking
    temperature=0.5,    # Less creative
    system_prompt="Be concise."
)
```

### Preset Configuration

```python
from backend.services.preset_registry import PresetRegistry
from backend.services.rag_pipeline import RAGConfig

registry = PresetRegistry()

# Load preset
config = registry.get_preset("research")

# Or build custom config
config = RAGConfig(
    top_k_retrieve=20,
    top_k_rerank=5,
    chunk_size=1000,
    overlap_size=200,
    temperature=0.7,
    system_prompt="You are an expert. Be detailed."
)

rag = RAGPipeline(
    chunk_store_path="data/chunks.jsonl",
    vector_store_path="data/vector_store",
    preset_registry=registry,
    config=config
)
```

### Presets YAML Format

backend/config/presets/simple.yaml
```yaml
top_k_retrieve: 5
top_k_rerank: 3
temperature: 0.5
system_prompt: "Answer directly and concisely."
```

backend/config/presets/research.yaml
```yaml
top_k_retrieve: 25
top_k_rerank: 10
temperature: 0.7
system_prompt: "Provide comprehensive, well-sourced answers. Cite specific papers and sections."
```

## Project Structure

```
RAG/
├── backend/
│   ├── api/                    # FastAPI backend (14 endpoints)
│   │   ├── main.py            # FastAPI app + lifespan
│   │   ├── dependencies.py     # Dependency injection
│   │   ├── schemas.py          # Pydantic models
│   │   ├── routes_query.py     # Query endpoints
│   │   ├── routes_documents.py # Document management
│   │   └── routes_system.py    # System status
│   │
│   ├── services/               # Core services (11 modules)
│   │   ├── document_loader.py
│   │   ├── chunking_service.py
│   │   ├── chunk_store.py
│   │   ├── paper_registry.py
│   │   ├── paper_processor.py
│   │   ├── embedding.py
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   ├── query_transformer.py
│   │   ├── reranker.py
│   │   ├── generation.py
│   │   └── rag_pipeline.py
│   │
│   ├── config/
│   │   └── presets/           # YAML configurations
│   │
│   ├── utils/
│   │   └── path_utils.py      # Security (relative paths)
│   │
│   └── init_rag.py            # CLI initialization script
│
├── data/
│   ├── raw_papers/            # Input PDFs
│   ├── chunks.jsonl           # Persisted chunks
│   ├── papers_manifest.json   # Processing metadata
│   └── vector_store/          # Chroma DB
│
├── examples.ipynb             # Interactive examples
├── test_api.sh               # API testing script
├── requirements.txt          # Dependencies
├── Dockerfile                # Container image
├── .dockerignore
├── .env.example              # Environment template
├── API.md                    # Full API documentation
├── ARCHITECTURE.md           # Detailed architecture
└── README.md                 # This file
```


## Next Steps (Optional)

These are post-MVP enhancements:

1. **Frontend** - React dashboard for document management
2. **Authentication** - JWT tokens for API security
3. **Caching** - Redis for query results
4. **Monitoring** - Prometheus metrics + Grafana
5. **Multi-user** - User accounts and document permissions
6. **Analytics** - Track query patterns and effectiveness
7. **Multi-modal** - Support for images and audio

## Troubleshooting

### GROQ_API_KEY not found
Ensure `.env` file exists in project root with:
```
GROQ_API_KEY=your_key_here
```

### No chunks found after processing
Check that:
1. PDFs are in `data/raw_papers/`
2. PDFs are readable (not corrupted)
3. Run: `python -m backend.init_rag` to reprocess

### API won't start
- Check Python version: 3.9+
- Verify all dependencies: `pip install -r requirements.txt`
- Check port 8000 is available
- Review logs in console

### Slow queries
- First query initializes RAGPipeline (~5 seconds)
- Subsequent queries are <1 second
- To speed up: reduce `top_k_retrieve` or `top_k_rerank`

## License

MIT - use it at your own risk da!

## Contact

Built with hehe by [Aaradhya Kulkarni](https://github.com/adhi1911)

---

Last updated: March 2026
