# PaperFlow

A minimal RAG (Retrieval-Augmented Generation) system for research papers.

## Overview

PaperFlow demonstrates a complete RAG pipeline:
1. **Load** research papers (PDFs) incrementally
2. **Chunk** semantically with overlap
3. **Index** with hybrid retrieval (BM25 + vector search)
4. **Rerank** for quality
5. **Generate** answers using GroqLLM

## Architecture

### Modular Services

Each service is independent, callable from Python or FastAPI later:

- **PaperRegistry**: Tracks which papers are processed via manifest file (detects duplicates/updates)
- **IncrementalChunker**: Chunks specific papers (reusable)
- **ChunkStore**: Manages chunks.jsonl persistence (append/remove/query)
- **PaperProcessor**: Orchestrates the workflow (coordinator)

### Incremental Processing

Only processes NEW and UPDATED papers:
```
First run:
  paper1.pdf + paper2.pdf → 350 chunks

Add paper3.pdf later:
  paper3.pdf only → 120 chunks (appended)
  Total: 470 chunks (no reprocessing)

Update paper1.pdf:
  Removes old chunks + adds new chunks
```

## Structure

```
backend/
├── services/
│   ├── paper_registry.py       # Tracks processed papers (manifest)
│   ├── incremental_chunker.py  # Chunks specific papers
│   ├── chunk_store.py          # Manages chunks.jsonl
│   ├── paper_processor.py      # Orchestrates workflow
│   ├── document_loader.py      # PDF loading
│   └── chunking_service.py     # Text chunking logic
├── core/
│   └── config.py               # Paths, chunk sizes
├── utils/
│   └── text_utils.py           # Text preprocessing
└── process_papers.py           # Main entry point

data/
├── raw_papers/                 # Place PDFs here (any time)
├── papers_manifest.json        # Auto-generated (tracks papers)
└── processed/
    └── chunks.jsonl            # All chunks (auto-generated)
```

## Quick Start

### Setup

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Create data directory
mkdir -p data/raw_papers
```

### Add Papers

Place PDF files in `data/raw_papers/`:
```bash
cp my_research_papers/*.pdf data/raw_papers/
```

### Process Papers

```bash
python -m backend.process_papers
```

**Output:**
- `data/raw_papers/papers_manifest.json` - Tracks which papers processed
- `data/processed/chunks.jsonl` - All chunks from all papers

### Add More Papers Later

Simply drop new PDFs in `data/raw_papers/` and run again:
```bash
cp new_paper.pdf data/raw_papers/
python -m backend.process_papers
```

Only the new paper is processed and chunks are appended.

### Check Status

```python
from backend.services.paper_processor import PaperProcessor
from backend.core.config import RAW_PAPERS_DIR, PROCESSED_CHUNKS_DIR

processor = PaperProcessor(
    papers_dir=RAW_PAPERS_DIR,
    manifest_path=RAW_PAPERS_DIR.parent / "papers_manifest.json",
    chunks_file=PROCESSED_CHUNKS_DIR / "chunks.jsonl",
)

status = processor.get_status()
print(status)
```

## Design Decisions

1. **Update handling**: Re-processed papers replace old chunks (clean, not versioned)
2. **Directory structure**: Flat (no subdirectories) for MVP
3. **Deletion**: Deferred - chunks remain immutable for now
4. **Service style**: Pure services, no web coupling (callable from FastAPI later)

## Development Roadmap

- [ ] Document loading & chunking
- [ ] Incremental paper processing
- [ ] Embedding generation (Qdrant)
- [ ] BM25 indexing
- [ ] Hybrid retrieval pipeline
- [ ] Reranking
- [ ] GroqLLM generation
- [ ] FastAPI endpoints
- [ ] React frontend
- [ ] Evaluation suite

## Technologies

- **PDF Processing**: PyMuPDF
- **LLM**: GroqLLM (Mixtral, Llama)
- **Embeddings**: Sentence-Transformers
- **Vector DB**: Qdrant
- **Backend**: FastAPI
- **Frontend**: React

---

**Type**: Portfolio Project | RAG System

--- 
Developed by Aaradhya Kulkarni | [GitHub](https://github.com/adhi1911)
