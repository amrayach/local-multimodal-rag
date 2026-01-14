# Local Page-First Multimodal RAG

A fully local multimodal Retrieval-Augmented Generation (RAG) system that processes PDF documents as page images and answers questions using visual language models.

## 🎯 Overview

This system implements a **page-first** approach to document Q&A:

1. **PDF → Page Images**: Each PDF page is rendered as a high-resolution image
2. **Visual Embeddings**: Pages are embedded using CLIP (upgradeable to ColPali/ColQwen2)
3. **Semantic Retrieval**: Questions are matched to relevant pages via FAISS vector search
4. **VLM Answering**: A vision-language model generates answers grounded in retrieved page images

This approach preserves visual layout, tables, charts, and formatting that traditional text-based RAG systems lose.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio UI (:8081)                        │
│                     ui/gradio_app.py                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────────┐
│                     FastAPI Gateway (:3001)                     │
│                        app/api.py                               │
│                                                                 │
│  POST /ingest  ─►  Upload & index PDF                          │
│  POST /chat    ─►  Question answering                          │
│  GET  /health  ─►  Health check                                │
│  GET  /stats   ─►  System statistics & observability           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     MMRagPipeline                               │
│                   app/rag/pipeline.py                           │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ PDF → Pages │  │  Embedder   │  │     FAISS Index         │ │
│  │ (PyMuPDF)   │  │   (CLIP)    │  │  (Cosine Similarity)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    VLM Answerer                          │   │
│  │              (Qwen2.5-VL - stub for now)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      Data Storage                               │
│                    data/mmrag/                                  │
│                                                                 │
│  docs/<doc_id>/                                                 │
│    ├── original.pdf      # Original uploaded PDF                │
│    ├── manifest.json     # Ingestion metadata & indexed flag    │
│    └── pages/            # Rendered page images                 │
│        ├── page_0001.png                                        │
│        ├── page_0002.png                                        │
│        └── ...                                                  │
│                                                                 │
│  index/                                                         │
│    ├── pages.faiss       # FAISS vector index                   │
│    └── pages.meta.json   # Page references metadata             │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
local_multimodal_rag/
├── app/
│   ├── api.py                 # FastAPI REST gateway
│   └── rag/
│       ├── __init__.py
│       ├── config.py          # Settings and paths
│       ├── storage.py         # File/directory management, manifests
│       ├── pdf_pages.py       # PDF to image conversion
│       ├── embedder.py        # CLIP-based page/text embeddings
│       ├── index_faiss.py     # FAISS index wrapper
│       ├── vlm_qwen25vl.py    # VLM answering (stub)
│       └── pipeline.py        # Main orchestration
├── ui/
│   └── gradio_app.py          # Gradio web interface
├── scripts/
│   ├── run_api.sh             # Start API server
│   ├── run_ui.sh              # Start Gradio UI
│   ├── reset_index.sh         # Clear FAISS index (keep docs)
│   └── reindex_all.sh         # Rebuild index from all docs
├── data/
│   └── mmrag/                 # Runtime data (created automatically)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
./scripts/run_api.sh
```

The API will be available at `http://localhost:3001`.

### 3. Start the UI (in a separate terminal)

```bash
./scripts/run_ui.sh
```

The Gradio UI will be available at `http://localhost:8081`.

### 4. Usage

1. Open the Gradio UI in your browser
2. Upload a PDF document (recommended: 30-40 pages)
3. Wait for ingestion to complete
4. Ask questions about the document
5. View the answer along with evidence pages

> **Demo Limits:** Max 100 pages per document, max 50MB file size, max 180 DPI rendering.

## 🔌 API Endpoints

### Health Check

```bash
curl http://localhost:3001/health
```

Response:
```json
{"ok": true, "indexed_pages": 42}
```

### System Stats

```bash
curl http://localhost:3001/stats
```

Response:
```json
{
  "num_docs": 3,
  "num_indexed_pages": 142,
  "embed_dim": 512,
  "embedder_id": "openai/clip-vit-base-patch32",
  "index_backend": "faiss",
  "index_type": "IndexFlatIP",
  "device": "cuda",
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_mb": 1024.5,
  "process_memory_mb": 2048.0,
  "demo_limits": {
    "max_pages": 100,
    "max_file_size_mb": 50,
    "max_dpi": 180
  }
}
```

### Ingest PDF

```bash
curl -X POST http://localhost:3001/ingest \
  -F "file=@document.pdf"
```

Response:
```json
{
  "doc_id": "a1b2c3d4e5f67890",
  "num_pages": 35,
  "is_new": true
}
```

### Chat / Ask Questions

```bash
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 3}'
```

Response:
```json
{
  "answer": "Based on the document...",
  "evidence": [
    {
      "doc_id": "a1b2c3d4e5f67890",
      "page_num": 12,
      "image_path": "/path/to/page_0012.png",
      "score": 0.8542
    }
  ]
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `3001` | API server port |
| `UI_HOST` | `0.0.0.0` | Gradio UI bind address |
| `UI_PORT` | `8081` | Gradio UI port |
| `API_BASE` | `http://127.0.0.1:3001` | API URL for UI to connect |

### Settings (app/rag/config.py)

- `base_dir`: Root data directory (`data/mmrag/`)
- `docs_dir`: Document storage
- `index_dir`: FAISS index storage
- `cache_dir`: Temporary cache

## 🔧 Components

### PDF Processing (`pdf_pages.py`)

- Uses PyMuPDF (fitz) for PDF rendering
- Default DPI: 180 (good balance of quality vs size)
- Output: PNG images per page

### Embedder (`embedder.py`)

- Default model: `openai/clip-vit-base-patch32`
- Supports batched image embedding
- L2-normalized vectors for cosine similarity

### FAISS Index (`index_faiss.py`)

- `IndexFlatIP` for inner product (cosine sim on normalized vectors)
- Metadata stored separately in JSON
- Supports incremental additions

### Manifest System (`storage.py`)

Each document has a `manifest.json`:
```json
{
  "doc_id": "a1b2c3d4e5f67890",
  "filename": "report.pdf",
  "num_pages": 35,
  "indexed": true,
  "created_at": "2025-12-15T10:30:00",
  "indexed_at": "2025-12-15T10:30:05",
  "sha256": "7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "index_backend": "faiss",
  "embedder_id": "openai/clip-vit-base-patch32"
}
```

This prevents duplicate indexing when re-uploading the same PDF. The `sha256` enables content verification and the `embedder_id` tracks which model was used for indexing.

## 🗺️ Roadmap

- [ ] **Replace CLIP with ColPali/ColQwen2** - Better document understanding
- [ ] **Implement Qwen2.5-VL answering** - Replace stub with actual VLM inference
- [ ] **Multi-document support** - Filter queries by document
- [ ] **Streaming responses** - Stream VLM output to UI
- [ ] **GPU acceleration** - Optimize for CUDA
- [ ] **Document deletion** - Remove documents from index
- [ ] **Hybrid retrieval** - Combine visual + text embeddings

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `gradio` | Web UI |
| `pymupdf` | PDF processing |
| `pillow` | Image handling |
| `torch` | Deep learning runtime |
| `transformers` | CLIP/VLM models |
| `faiss-cpu` | Vector similarity search |
| `numpy` | Numerical operations |

## 🐛 Troubleshooting

### API not reachable from UI

Ensure the API is running before starting the UI:
```bash
# Terminal 1
./scripts/run_api.sh

# Terminal 2 (after API is up)
./scripts/run_ui.sh
```

### Out of memory during embedding

Reduce batch size in `embedder.py`:
```python
def embed_images(self, image_paths: list[Path], batch_size: int = 8):  # Lower from 16
```

### Slow PDF processing

Lower DPI in `pipeline.py`:
```python
page_imgs = pdf_to_page_images(pdf_path, pdir, dpi=144)  # Lower from 180
```

### Re-indexing a document

Delete the manifest to force re-indexing:
```bash
rm data/mmrag/docs/<doc_id>/manifest.json
```

### Reset and rebuild entire index

```bash
# Clear the index (keeps document files)
./scripts/reset_index.sh

# Rebuild from all existing documents
./scripts/reindex_all.sh
```

### Check system stats

```bash
curl http://localhost:3001/stats | jq .
```

## 📄 License

MIT License - See LICENSE file for details.
