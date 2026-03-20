# PaperFlow

End-to-end RAG initialization: Load PDFs → chunk → embed → semantic search. Ready for LLM generation.

## Features

- **PDF to embeddings**: Auto-pipeline from raw PDFs to dense vectors
- **LangChain-native**: Full `Document` objects preserved with metadata throughout
- **Semantic search**: Query by natural language, get similar chunks ranked by cosine similarity
- **Incremental processing**: Only reprocess modified papers (MD5 hashing)
- **Flexible storage**: JSONL chunks + Chroma vector DB (easy swap to Milvus/Qdrant)
- **Change tracking**: Auto-detects new/modified papers, auto-deletes old embeddings

## Quick Start

### 1. Install
```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Configure
Create `.env`:
```
RAW_PAPERS_DIR=data/raw_papers
CHUNKS_FILE=data/chunks.jsonl
MANIFEST_PATH=data/papers_manifest.json
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 3. Process Papers
```bash
# Add PDFs to data/raw_papers/
cp /path/to/papers/*.pdf data/raw_papers/

# Process
python -m backend.init_rag
```

### 4. Query
```python
from backend.services.chunk_store import ChunkStore
from backend.services.settings import settings

store = ChunkStore(settings.chunks_file)

# Get all chunks
all_chunks = store.load_all()

# Get chunks from specific paper
paper_chunks = store.get_by_source("research_paper.pdf")

# Get stats
stats = store.get_stats()  # {total_chunks, total_sources, chunks_per_source}
```

## API Reference

### DocumentLoader
```python
from backend.services.document_loader import DocumentLoader

loader = DocumentLoader()
docs = loader.load_all_pdfs()           # Bulk load from data/raw_papers/
docs = loader.load_specific_pdf("name.pdf")  # Single file
```

### ChunkingService
```python
from backend.services.chunking_service import ChunkingService

chunker = ChunkingService()
chunks = chunker.chunk_documents(docs)  # Returns Document objects with chunk_index
```

### ChunkStore
```python
from backend.services.chunk_store import ChunkStore

store = ChunkStore("data/chunks.jsonl")
store.append_chunks(chunks)              # Add chunks
all_chunks = store.load_all()            # Get all chunks
paper_chunks = store.get_by_source("file.pdf")
store.remove_by_source("file.pdf")       # Delete by paper
stats = store.get_stats()
```

### PaperRegistry
```python
from backend.services.paper_registry import PaperRegistry

registry = PaperRegistry("data/papers_manifest.json")
new = registry.get_new_papers("data/raw_papers/")      # Unprocessed
updated = registry.get_updated_papers("data/raw_papers/")  # Modified
status = registry.get_status()           # {total_papers, total_chunks, papers: [...]}
```

### PaperProcessor (Orchestrator)
```python
from backend.services.paper_processor import PaperProcessor

processor = PaperProcessor()
result = processor.process_all()  # Load → chunk → store → track
# Returns: {new_processed, updated_processed, total_chunks, status}
```

### EmbeddingService
```python
from backend.services.embedding import EmbeddingService

embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
embed_texts = embedding_service.embed_texts(texts)  # Returns np.ndarray
results = embedding_service.embed_chunks(chunks)     # Returns list[{chunk_id, embedding, metadata}]
```

### VectorStore (Chroma)
```python
from backend.services.vector_store import VectorStore

vector_store = VectorStore()
vector_store.add_embeddings(embedding_results)       # Add from EmbeddingService
results = vector_store.search(query_embedding, top_k=5, score_threshold=0.0)
vector_store.delete_by_source("paper.pdf")          # Remove old vectors
stats = vector_store.get_stats()                     # {total_vectors, total_sources}
```

## How It Works

```
PDFs (data/raw_papers/)
  ↓
DocumentLoader (DirectoryLoader bulk + PyMuPDFLoader single)
  ↓
LangChain Documents (source, page metadata)
  ↓
ChunkingService (RecursiveCharacterTextSplitter: 1000 char, 200 overlap)
  ↓
Chunks with chunk_index metadata
  ↓
ChunkStore (JSONL persistence, queryable by source/page)
  ↓
PaperRegistry (MD5 tracking, detects new/modified papers)
  ↓
EmbeddingService (Sentence-Transformers: all-MiniLM-L6-v2, 384-dim)
  ↓
VectorStore (Chroma: stores embeddings + metadata, enables ANN search)
  ↓
Semantic Search (cosine similarity, ranked results)
```

Each service is independent—use them individually or orchestrate with `PaperProcessor`.

## Examples

See [examples.ipynb](examples.ipynb) for a working 9-step notebook:
1. Setup (imports, config)
2. Check prerequisites
3. Process papers → chunks
4. Inspect results
5. Query chunks
6. Query by paper
7. Detect changes
8. Generate embeddings (384-dim)
9. Semantic search


## Tech Stack

**Document Pipeline:**
- **PDF Processing**: PyMuPDF
- **Document Format**: LangChain Document objects
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Chunk Storage**: JSONL (preserves metadata)
- **Change Detection**: MD5 hashing

**Embedding & Search:**
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2: 384-dim, fast)
- **Vector DB**: Chroma (local, persistent, easy to scale to Milvus/Qdrant)
- **Search**: ANN via Chroma (cosine similarity, ranked by score)

**Core:**
- **Config**: Pydantic Settings + .env

## Development Status

**✅ Completed:**
- Document loading (18 PDFs)
- Chunking (1387 chunks, 1000 char / 200 overlap)
- Chunk storage (JSONL format with metadata)
- Incremental processing (MD5 tracking)
- Embedding generation (384-dim Sentence-Transformers)
- Vector storage (Chroma, semantic search ready)

**📌 Next Phase - Retrieval & Generation:**
- [ ] Query transformation (HyDE, multi-query)
- [ ] BM25 for hybrid retrieval (dense + sparse)
- [ ] Re-ranking (cross-encoder)
- [ ] GroqLLM integration (generation)
- [ ] RAG pipeline: query → retrieve → generate
- [ ] FastAPI + React UI

## License

MIT
