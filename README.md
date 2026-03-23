# PaperFlow - RAG Done Right

Turn your PDF mess into a production-ready RAG system. Load papers -> chunk -> embed -> semantic search -> LLM answers. No BS, just clean modular code.

## What You Get

- **PDF to embeddings** - Automated pipeline from raw PDFs to searchable vectors (384-dim)
- **Semantic search** - Query in English, get ranked results by relevance (cosine similarity)
- **Smart incremental processing** - Only reprocess changed papers (MD5 hashing), skip the rest
- **LangChain-native** - Document objects with metadata flow through entire pipeline
- **Flexible storage** - JSONL chunks + Chroma vector DB (easily swap to Milvus/Qdrant later)
- **Change tracking** - Auto-detects new/modified papers, cleans up old embeddings

## Architecture Overview

Composed of 11 modular services. Use solo or compose together.

    PDFs (raw_papers/)
        |
        v
    DocumentLoader -> ChunkingService -> ChunkStore -> PaperRegistry
        |
        v
    EmbeddingService -> VectorStore -> RAGPipeline
        |
        v
    HybridRetriever -> QueryTransformer -> CrossEncoderReranker -> GroqGenerator

Each service does ONE thing well. Mix and match.

## How It Works

1. **Load** PDFs from disk
2. **Chunk** while preserving metadata (1000 chars, 200 overlap)
3. **Store** chunks in JSONL format
4. **Track** papers via MD5 hashing (incremental processing)
5. **Embed** using Sentence-Transformers (384-dim)
6. **Search** using Chroma vector DB (semantic + keyword via hybrid retrieval)
7. **Rerank** using cross-encoder for accuracy
8. **Generate** answers using Groq LLM with flexible system prompts

## Quick Start (5 mins)

### Installation

```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Setup

Create `.env` in project root:
```
GROQ_API_KEY=your_key_here
```

### Process PDFs

```bash
# Copy PDFs to data/raw_papers/
cp /path/to/papers/*.pdf data/raw_papers/

# Run pipeline: load -> chunk -> store -> embed
python -m backend.init_rag
```

### Start Querying

Text-based queries work immediately after setup.

## Design Patterns

>> Config-as-Code 
Behavior lives in YAML presets, not hardcoded. Change retrieval strategy, LLM settings, or reranking without touching code.

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

>> Phase 1: Document Pipeline - Complete
  18 research papers, 1,387 chunks, all embedded (384-dim), MD5 incremental processing working.

>> Phase 2: Retrieval & Generation - Complete & Tested
  Query expansion + hybrid retrieval + reranking + LLM generation. RAGPipeline orchestrator. YAML presets. Full pipeline tested in examples.ipynb.

>> Phase 3: Deployment - Next
  FastAPI backend, React frontend, Docker.

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

---

Built with clean architecture. Deploy with confidence.
