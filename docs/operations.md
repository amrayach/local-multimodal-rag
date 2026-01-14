# Operations Guide

This guide covers operational tasks for managing the Local Multimodal RAG system.

## Index Management

### Reset Index

Clear the FAISS index while preserving all document files:

```bash
./scripts/reset_index.sh
```

This script:
1. Removes `data/mmrag/index/` directory (FAISS index + metadata)
2. Updates all `manifest.json` files to set `indexed=false`
3. Preserves original PDFs and rendered page images

**Use case:** When you want to rebuild the index from scratch, switch embedder models, or fix a corrupted index.

### Rebuild Index

Reindex all documents from scratch:

```bash
./scripts/reindex_all.sh
```

This script:
1. Clears the existing index
2. Walks all documents in `data/mmrag/docs/`
3. Re-embeds all pages and rebuilds the FAISS index
4. Updates all manifests with current embedder info

**Output example:**
```
🔄 Reindexing all documents...
Pipeline initialized with 0 indexed pages
Index cleared
Rendering pages for a1b2c3d4e5f67890
Reindexed a1b2c3d4e5f67890: 35 pages
Reindexed 7f83b1657ff1fc53: 42 pages

✅ Reindex complete:
   Documents: 2
   Pages: 77
   Elapsed: 15234.5ms
```

---

## Monitoring & Observability

### Health Check

Quick endpoint to verify the API is running:

```bash
curl http://localhost:3001/health
```

Response:
```json
{"ok": true, "indexed_pages": 142}
```

### System Statistics

Detailed system stats for monitoring dashboards:

```bash
curl http://localhost:3001/stats | jq .
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

### Structured Logs

The API emits structured logs for each ingest operation:

```
2025-12-15 10:30:05 | INFO | app.rag.pipeline | Indexed 35 pages for doc a1b2c3d4e5f67890 | timings: render=2345.6ms, embed=1234.5ms, index=45.2ms, total=3625.3ms
```

Log fields:
- `render`: Time to convert PDF to images (ms)
- `embed`: Time to generate embeddings (ms)
- `index`: Time to add vectors to FAISS (ms)
- `total`: Total ingestion time (ms)

---

## Demo Limits

The system enforces the following limits for demo stability:

| Limit | Value | Configurable |
|-------|-------|--------------|
| Max pages per document | 100 | `app/rag/pipeline.py:MAX_PAGES` |
| Max file size | 50 MB | `app/rag/pipeline.py:MAX_FILE_SIZE_MB` |
| Max render DPI | 180 | `app/rag/pipeline.py:MAX_DPI` |

### Error Messages

When limits are exceeded, the API returns clear error messages:

```json
{"detail": "File too large: 55.2MB exceeds 50MB limit"}
{"detail": "Too many pages: 150 exceeds 100 page limit"}
```

---

## Idempotent Ingestion

The system uses content-addressable storage to prevent duplicate indexing:

1. **Document ID**: First 16 characters of SHA-256 hash of file content
2. **Manifest Check**: Before processing, checks if `manifest.indexed=true`
3. **Skip Logic**: If already indexed, returns cached result without re-embedding

### Manifest Structure

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

Fields:
- `sha256`: Full content hash for verification
- `index_backend`: Vector store type used
- `embedder_id`: Model used for embeddings (important for reindexing)

---

## Data Directory Structure

```
data/mmrag/
├── docs/
│   └── <doc_id>/
│       ├── original.pdf      # Original uploaded file
│       ├── manifest.json     # Metadata & indexing state
│       └── pages/
│           ├── page_0001.png
│           ├── page_0002.png
│           └── ...
├── index/
│   ├── pages.faiss           # FAISS vector index
│   └── pages.meta.json       # Page reference metadata
└── cache/                    # Temporary files (safe to delete)
```

### Manual Operations

**Force re-index a single document:**
```bash
# Remove manifest to trigger re-indexing on next upload
rm data/mmrag/docs/<doc_id>/manifest.json
```

**Delete a document completely:**
```bash
rm -rf data/mmrag/docs/<doc_id>
# Then run reindex to update FAISS
./scripts/reindex_all.sh
```

**Check document count:**
```bash
ls -1 data/mmrag/docs | wc -l
```

**Check total pages:**
```bash
find data/mmrag/docs -name "page_*.png" | wc -l
```

---

## Troubleshooting

### Index Corruption

If the FAISS index becomes corrupted or out of sync:

```bash
./scripts/reset_index.sh
./scripts/reindex_all.sh
```

### Out of Memory

Reduce batch size in `app/rag/embedder.py`:
```python
def embed_images(self, image_paths: list[Path], batch_size: int = 8):  # Lower from 16
```

### Slow Ingestion

1. Check GPU availability: `curl localhost:3001/stats | jq .device`
2. Lower DPI for faster rendering (edit `MAX_DPI` in pipeline.py)
3. Ensure SSD storage for `data/mmrag/`

### API Not Reachable

```bash
# Check if process is running
pgrep -f "uvicorn app.api"

# Check port binding
ss -tlnp | grep 3001

# Restart API
./scripts/run_api.sh
```
