# PaperFlow

A minimal RAG (Retrieval-Augmented Generation) system for research papers using **LangChain** primitives.

## ⭐ Key Features

- **LangChain-based**: Uses LangChain `Document` objects throughout pipeline
- **Modular services**: Each service independent and reusable
- **Incremental processing**: Only processes new/updated papers
- **Persistent chunks**: Stores chunks with metadata in JSONL
- **No config overhead**: Simple directory-based setup

## Architecture

### Core Data Flow

```
PDFs (any time)
    ↓
DocumentLoader (DirectoryLoader + PyMuPDFLoader)
    → LangChain Document objects
    ↓
ChunkingService (RecursiveCharacterTextSplitter)
    → LangChain Document chunks with metadata
    ↓
ChunkStore
    → Persisted to chunks.jsonl
```

### Services (LangChain Document-based)

**DocumentLoader** - Load PDFs as LangChain Documents
- Uses: `langchain_community.document_loaders.DirectoryLoader` + `PyMuPDFLoader`
- Returns: `List[Document]` with metadata (source, page)

**ChunkingService** - Split Documents preserving metadata
- Uses: `langchain_text_splitters.RecursiveCharacterTextSplitter`
- Config: chunk_size=1000, chunk_overlap=200
- Returns: `List[Document]` with chunk_index metadata

**ChunkStore** - Persist and query Document chunks
- Serializes: Document ↔ JSON with full metadata preservation
- Methods: `load_all_chunks()`, `append_chunks()`, `get_chunks_by_source()`, `remove_chunks_by_source()`
- Returns: `List[Document]`

**IncrementalChunker** - Chunk specific papers
- Orchestrates: DocumentLoader → ChunkingService
- Returns: `List[Document]`

**PaperRegistry** - Track processed papers
- Detects: New papers (not in manifest) and updated papers (hash changed)
- Persists: papers_manifest.json with MD5 hashes
- Purpose: Avoid reprocessing

**PaperProcessor** - Full orchestration
- Workflow:
  1. Registry finds new/updated papers
  2. IncrementalChunker processes them
  3. ChunkStore persists/updates chunks
  4. Registry updates manifest
- Returns: Processing summary

## Directory Structure

```
backend/
├── services/
│   ├── document_loader.py       # LangChain DirectoryLoader wrapper
│   ├── chunking_service.py      # LangChain RecursiveCharacterTextSplitter wrapper
│   ├── incremental_chunker.py   # Orchestrates loader + chunker
│   ├── chunk_store.py           # Document serialization + JSONL persistence
│   ├── paper_registry.py        # Manifest-based change detection
│   └── paper_processor.py       # Full orchestration
├── process_papers.py            # CLI entry point
├── example_usage.py             # Service examples
├── test_services.py             # Unit tests
└── requirements.txt             # Dependencies

data/
├── raw_papers/                  # Add PDF files here
├── papers_manifest.json         # Auto-generated (tracks papers)
└── chunks.jsonl                 # Auto-generated (all chunks)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

Dependencies include:
- `langchain-core` - Document objects
- `langchain-community` - DirectoryLoader, PyMuPDFLoader
- `langchain-text-splitters` - RecursiveCharacterTextSplitter
- `pymupdf` - PDF loading

### 2. Add Papers

Place PDF files in `data/raw_papers/`:
```bash
mkdir -p data/raw_papers
cp my_research_papers/*.pdf data/raw_papers/
```

### 3. Process Papers

```bash
python -m backend.process_papers
```

**First Run Output:**
```
Started paper processing...
New papers: 3
Updated papers: 0
Processing complete! ✓
Total chunks: 427
```

**Next Run (with new papers):**
```
Started paper processing...
New papers: 1
Updated papers: 0
Processing complete! ✓
Total chunks: 523  # Only new paper chunks added
```

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
