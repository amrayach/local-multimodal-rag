# Architecture Documentation

## System Overview

The Local Multimodal RAG system is a **page-first** document question-answering pipeline. Unlike traditional RAG systems that extract and chunk text, this system treats each PDF page as an atomic visual unit, preserving layout, tables, figures, and formatting.

## Design Principles

1. **Visual Fidelity** - Pages are processed as images, not text
2. **Content Addressability** - Documents are identified by content hash (SHA-256)
3. **Idempotent Ingestion** - Re-uploading the same PDF is a no-op
4. **Separation of Concerns** - API, UI, and RAG logic are decoupled
5. **Lazy Loading** - Heavy models load only when needed

## Component Architecture

### Layer Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                        │
│  ┌─────────────────────┐    ┌─────────────────────────────┐   │
│  │   Gradio UI         │    │   FastAPI REST API          │   │
│  │   (ui/gradio_app)   │───►│   (app/api)                 │   │
│  └─────────────────────┘    └──────────────┬──────────────┘   │
└────────────────────────────────────────────┼──────────────────┘
                                             │
┌────────────────────────────────────────────▼──────────────────┐
│                      Business Logic Layer                      │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                    MMRagPipeline                        │   │
│  │                  (app/rag/pipeline)                     │   │
│  └────────────────────────────────────────────────────────┘   │
│                              │                                 │
│  ┌──────────────┬────────────┼────────────┬───────────────┐   │
│  ▼              ▼            ▼            ▼               ▼   │
│ PDF→Images   Embedder    FaissIndex   VLMAnswerer   Storage   │
│ (pdf_pages)  (embedder)  (index_faiss) (vlm_qwen)   (storage) │
└────────────────────────────────────────────────────────────────┘
                                             │
┌────────────────────────────────────────────▼──────────────────┐
│                        Data Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐   │
│  │  Document Store │  │  Vector Index   │  │    Cache     │   │
│  │  (docs/)        │  │  (index/)       │  │  (cache/)    │   │
│  └─────────────────┘  └─────────────────┘  └──────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `config.py` | Centralized settings, paths | None |
| `storage.py` | File I/O, manifest management | config |
| `pdf_pages.py` | PDF → PNG conversion | PyMuPDF, PIL |
| `embedder.py` | Visual/text embeddings | transformers, torch |
| `index_faiss.py` | Vector storage & retrieval | faiss, numpy |
| `vlm_qwen25vl.py` | Answer generation | transformers (stub) |
| `pipeline.py` | Orchestration | All above |
| `api.py` | HTTP interface | pipeline, FastAPI |
| `gradio_app.py` | Web UI | requests, Gradio |

## Data Flow

### Ingestion Flow

```
PDF Bytes
    │
    ▼
┌─────────────────────┐
│ doc_id = SHA256[:16]│  Content-addressable ID
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check manifest.json │  Already indexed?
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │ indexed=true│───► Return cached result
    └──────┬──────┘
           │ indexed=false or missing
           ▼
┌─────────────────────┐
│ Save original.pdf   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Render page images  │  PyMuPDF @ 180 DPI
│ page_0001.png, ...  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Embed pages (CLIP)  │  Batched, normalized
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Add to FAISS index  │  IndexFlatIP
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Save manifest       │  indexed=true
│ indexed_at=timestamp│
└─────────────────────┘
```

### Query Flow

```
Question (string)
    │
    ▼
┌─────────────────────┐
│ Embed query (CLIP)  │  Text embedding
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FAISS search        │  Top-K nearest neighbors
│ (cosine similarity) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Load evidence pages │  PNG paths from metadata
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ VLM inference       │  Question + images → answer
│ (Qwen2.5-VL)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Return answer +     │
│ evidence references │
└─────────────────────┘
```

## Storage Schema

### Directory Structure

```
data/mmrag/
├── docs/
│   └── <doc_id>/              # 16-char hex hash
│       ├── original.pdf       # Immutable source
│       ├── manifest.json      # Metadata + state
│       └── pages/
│           ├── page_0001.png  # 1-indexed
│           ├── page_0002.png
│           └── ...
├── index/
│   ├── pages.faiss            # Binary FAISS index
│   └── pages.meta.json        # PageRef array
└── cache/                     # Reserved for future use
```

### Manifest Schema

```json
{
  "doc_id": "a1b2c3d4e5f67890",
  "filename": "annual_report.pdf",
  "num_pages": 42,
  "indexed": true,
  "created_at": "2025-12-15T10:30:00.000000",
  "indexed_at": "2025-12-15T10:30:15.000000",
  "sha256": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "index_backend": "faiss",
  "embedder_id": "openai/clip-vit-base-patch32"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | First 16 chars of SHA-256 hash |
| `filename` | string | Original upload filename |
| `num_pages` | integer | Page count |
| `indexed` | boolean | Whether vectors are in FAISS |
| `created_at` | ISO datetime | First upload time |
| `indexed_at` | ISO datetime | Last indexing time |
| `sha256` | string | Full SHA-256 hash for verification |
| `index_backend` | string | Vector store type (faiss) |
| `embedder_id` | string | Model used for embeddings |

### Index Metadata Schema

```json
[
  {
    "doc_id": "a1b2c3d4e5f67890",
    "page_num": 1,
    "image_path": "/path/to/data/mmrag/docs/a1b2c3d4e5f67890/pages/page_0001.png"
  },
  ...
]
```

## Vector Index Design

### Why FAISS IndexFlatIP?

- **IndexFlatIP** = Inner Product index (brute-force)
- With L2-normalized vectors, inner product = cosine similarity
- Exact search (no approximation)
- Suitable for <100K vectors; upgrade to IVF for larger scale

### Embedding Dimensions

| Model | Dimension | Notes |
|-------|-----------|-------|
| CLIP ViT-B/32 | 512 | Current default |
| CLIP ViT-L/14 | 768 | Higher quality |
| ColPali | 128 | Planned upgrade |

## API Design

### RESTful Conventions

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness + basic stats |
| `/stats` | GET | Detailed system statistics |
| `/ingest` | POST | Upload PDF (multipart) |
| `/chat` | POST | Question answering (JSON) |

### Demo Limits

| Limit | Value | Purpose |
|-------|-------|---------|
| Max pages | 100 | Prevent runaway ingestion |
| Max file size | 50 MB | Memory protection |
| Max DPI | 180 | Balance quality/speed |

### Error Handling

- `400 Bad Request` - Invalid input (non-PDF, empty file, exceeds limits)
- `500 Internal Server Error` - Processing failure
- Errors include descriptive messages in response body

### Observability

The `/stats` endpoint provides:
- Document and page counts
- Embedding dimension and model ID
- Device info (CPU/GPU, GPU name, memory)
- Demo limit configuration

Structured logs emit timing for each ingest stage:
- `pdf_render_ms`: PDF to image conversion
- `embed_ms`: Image embedding generation
- `index_add_ms`: FAISS index insertion
- `total_ms`: End-to-end ingestion time

## Concurrency Model

### Current (Single-threaded)

- Uvicorn runs single worker by default
- Pipeline initialized once at startup (lifespan)
- FAISS index is not thread-safe for writes

### Future (Multi-worker)

For production with multiple workers:
1. Use separate index per worker, or
2. Use FAISS with `faiss.IndexIDMap` + locking, or
3. Move to vector DB (Milvus, Qdrant)

## Memory Considerations

### Peak Memory Usage

| Operation | Estimated Memory |
|-----------|-----------------|
| CLIP model (ViT-B/32) | ~400 MB |
| 100-page PDF images | ~500 MB |
| Embedding 100 pages | ~200 MB (batch) |
| FAISS index (10K vectors) | ~20 MB |

### Optimization Strategies

1. **Batch embedding** - Process images in chunks of 16
2. **Lazy model loading** - Load models only when first used
3. **Image streaming** - Don't hold all pages in memory
4. **Lower DPI** - Reduce from 180 to 144 for memory savings
