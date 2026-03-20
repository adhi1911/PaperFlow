# PaperFlow

Load research papers, split into chunks, and store with full metadata. Build a knowledge base for RAG systems in 3 lines of code.

## Features

- **Simple API**: Load PDFs → chunk → query in 3 service calls
- **LangChain-native**: Full `Document` objects preserved throughout (metadata intact)
- **Incremental processing**: Only reprocess modified papers (MD5 hashing)
- **Flexible loading**: Bulk load via DirectoryLoader or single file via PyMuPDFLoader
- **Persistent storage**: JSONL format with queryable metadata
- **Change tracking**: Auto-detects new/modified papers, avoids redundant work

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

## How It Works

```
PDFs (data/raw_papers/)
  ↓
DocumentLoader (load_all_pdfs or load_specific_pdf)
  ↓
LangChain Documents with metadata (source, page)
  ↓
ChunkingService (RecursiveCharacterTextSplitter)
  ↓
Chunks with added chunk_index metadata
  ↓
ChunkStore (JSONL persistence)
  ↓
Query by source, get stats, track changes
  ↓
PaperRegistry (MD5 hashing for change detection)
```

Each service is independent—use them individually or orchestrate with `PaperProcessor`.

## Examples

See [examples.ipynb](examples.ipynb) for a working 7-step notebook:
1. Setup (imports, config)
2. Check prerequisites
3. Process papers
4. Inspect results
5. Query chunks
6. Query by paper
7. Detect changes


## Tech Stack

- **PDF Processing**: PyMuPDF
- **Document Format**: LangChain Document objects
- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Storage**: JSONL (preserves full metadata)
- **Change Detection**: MD5 hashing
- **Config**: Pydantic Settings + .env

## Next Steps

Build embeddings, vector search, and LLM generation on top of this knowledge base:
- [ ] Sentence-Transformers for embeddings
- [ ] Vector storage
- [ ] Hybrid retrieval (BM25 + semantic)
- [ ] GroqLLM integration
- [ ] FastAPI + React frontend

## License

MIT
