# Development Progress

## Project Timeline

### Phase 1: Project Setup ✅
**Date: December 15, 2025**

- [x] Created project skeleton structure
- [x] Set up virtual environment
- [x] Created `requirements.txt` with all dependencies
- [x] Installed packages (fastapi, gradio, torch, transformers, faiss, etc.)

### Phase 2: Core RAG Components ✅
**Date: December 15, 2025**

#### Configuration (`config.py`)
- [x] Implemented `Settings` dataclass with frozen=True
- [x] Used properties for derived paths (docs_dir, index_dir, cache_dir)
- [x] Centralized all path configuration

#### Storage (`storage.py`)
- [x] Implemented `ensure_dirs()` for directory creation
- [x] Implemented `doc_id_from_bytes()` using SHA-256
- [x] Added path helpers: `doc_dir()`, `pages_dir()`, `original_pdf_path()`
- [x] Added manifest system for idempotent ingestion:
  - `DocManifest` dataclass
  - `manifest_path()`, `load_manifest()`, `save_manifest()`

#### PDF Processing (`pdf_pages.py`)
- [x] Implemented `pdf_to_page_images()` using PyMuPDF
- [x] Configurable DPI (default 180)
- [x] PNG optimization enabled
- [x] Proper resource cleanup with context manager

#### Embedder (`embedder.py`)
- [x] Implemented `PageEmbedder` class using CLIP
- [x] Default model: `openai/clip-vit-base-patch32`
- [x] Batched image embedding with configurable batch_size
- [x] L2 normalization for cosine similarity
- [x] `@torch.inference_mode()` for efficiency

#### FAISS Index (`index_faiss.py`)
- [x] Implemented `PageRef` dataclass for metadata
- [x] Implemented `FaissPageIndex` class
- [x] `IndexFlatIP` for cosine similarity (on normalized vectors)
- [x] Separate metadata storage in JSON
- [x] Load/save persistence
- [x] `total_pages` property for monitoring

#### VLM Answerer (`vlm_qwen25vl.py`)
- [x] Created stub implementation
- [x] Prepared for Qwen2.5-VL integration
- [x] Image validation in place

#### Pipeline (`pipeline.py`)
- [x] Implemented `MMRagPipeline` orchestrator
- [x] `ingest_pdf_bytes()` - full ingestion flow
- [x] `chat()` - retrieval + answering flow
- [x] Manifest-based deduplication
- [x] Logging throughout

### Phase 3: API & UI ✅
**Date: December 15, 2025**

#### FastAPI Gateway (`api.py`)
- [x] Health check endpoint with index stats
- [x] Ingest endpoint with file validation
- [x] Chat endpoint with Pydantic models
- [x] Lifespan context manager for startup
- [x] Response models for OpenAPI docs

#### Gradio UI (`gradio_app.py`)
- [x] PDF upload with ingestion
- [x] Question input with top_k slider
- [x] Answer display
- [x] Evidence gallery with page images
- [x] API health check on load
- [x] Error handling for all operations
- [x] Environment variable configuration

#### Scripts
- [x] `run_api.sh` - API server launcher
- [x] `run_ui.sh` - UI launcher
- [x] Both made executable
- [x] Environment variable support

### Phase 4: Documentation ✅
**Date: December 15, 2025**

- [x] README.md with full documentation
- [x] Architecture documentation
- [x] Workflows documentation
- [x] API reference
- [x] Progress tracking (this file)

---

## Current Status

### Working Features
| Feature | Status | Notes |
|---------|--------|-------|
| PDF upload | ✅ Working | Via API and UI |
| Page rendering | ✅ Working | PyMuPDF @ 180 DPI |
| CLIP embedding | ✅ Working | Batched, normalized |
| FAISS indexing | ✅ Working | Cosine similarity |
| Semantic search | ✅ Working | Top-K retrieval |
| Evidence display | ✅ Working | Gallery in UI |
| Deduplication | ✅ Working | Manifest-based |
| VLM answering | 🔶 Stub | Returns placeholder |

### Known Limitations
1. **VLM is a stub** - Returns placeholder text, not actual Qwen2.5-VL inference
2. **Single document queries** - Can't filter by specific document
3. **No document deletion** - Must manually delete from filesystem
4. **No authentication** - API is open
5. **Single worker** - Not optimized for concurrent requests

---

## Next Steps

### Immediate (High Priority)
- [ ] Implement actual Qwen2.5-VL inference in `vlm_qwen25vl.py`
- [ ] Test with real PDF documents
- [ ] Add error handling for corrupted PDFs

### Short Term
- [ ] Add document listing endpoint (`GET /documents`)
- [ ] Add document deletion endpoint (`DELETE /documents/{doc_id}`)
- [ ] Filter chat by document ID
- [ ] Add progress indicators for long operations

### Medium Term
- [ ] Replace CLIP with ColPali/ColQwen2 for better retrieval
- [ ] Add streaming responses for VLM output
- [ ] Implement multi-worker support
- [ ] Add basic authentication

### Long Term
- [ ] Support multiple file types (DOCX, PPTX, images)
- [ ] Hybrid retrieval (visual + text embeddings)
- [ ] Fine-tune embeddings on domain data
- [ ] Add evaluation metrics

---

## Changelog

### 2025-12-15
- Initial project creation
- Implemented all core components
- Created API and UI
- Added manifest-based deduplication
- Created documentation

---

## Issues & Bugs

### Open
*None currently tracked*

### Resolved
| Issue | Resolution | Date |
|-------|------------|------|
| `python-multipart` version conflict | Updated to `>=0.0.18` | 2025-12-15 |

---

## Performance Benchmarks

*To be measured with real workloads*

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| PDF ingestion (40 pages) | ~30-60s | Depends on PDF complexity |
| Page embedding (40 pages) | ~10-20s | With batching |
| Query search | <100ms | FAISS brute-force |
| VLM inference | TBD | Depends on model |

---

## Dependencies Version Lock

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
python-multipart>=0.0.18
gradio==5.8.0
pillow==11.0.0
pymupdf==1.24.14
numpy==1.26.4
faiss-cpu==1.8.0.post1
torch (latest)
transformers (latest)
accelerate (latest)
```
