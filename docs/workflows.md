# Workflows Documentation

## Development Workflows

### Initial Setup

```bash
# 1. Clone or create project directory
cd /home/ammer/local_multimodal_rag

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### Running the System

#### Terminal 1: API Server

```bash
./scripts/run_api.sh

# Or with custom settings:
API_PORT=8000 ./scripts/run_api.sh
```

Expected output:
```
Starting MMRAG API on http://0.0.0.0:3001
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:3001
```

#### Terminal 2: Gradio UI

```bash
./scripts/run_ui.sh

# Or connect to different API:
API_BASE=http://192.168.1.100:3001 ./scripts/run_ui.sh
```

Expected output:
```
Starting MMRAG UI on http://0.0.0.0:8081
Connecting to API at http://127.0.0.1:3001
Running on local URL:  http://0.0.0.0:8081
```

### Testing the Pipeline

#### Via curl (API)

```bash
# Health check
curl http://localhost:3001/health

# Ingest a PDF
curl -X POST http://localhost:3001/ingest \
  -F "file=@/path/to/document.pdf"

# Ask a question
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 3}'
```

#### Via Python (Direct)

```python
import sys
sys.path.insert(0, '/home/ammer/local_multimodal_rag')

from app.rag.pipeline import MMRagPipeline

# Initialize pipeline
pipe = MMRagPipeline()

# Ingest a PDF
with open("document.pdf", "rb") as f:
    result = pipe.ingest_pdf_bytes(f.read(), "document.pdf")
print(result)

# Ask a question
response = pipe.chat("What are the key findings?", top_k=3)
print(response["answer"])
for e in response["evidence"]:
    print(f"  Page {e['page_num']}: {e['score']:.4f}")
```

## Data Management Workflows

### Inspecting Indexed Documents

```bash
# List all indexed documents
ls -la data/mmrag/docs/

# View a document's manifest
cat data/mmrag/docs/<doc_id>/manifest.json | jq .

# Count pages for a document
ls data/mmrag/docs/<doc_id>/pages/ | wc -l

# Check index size
ls -lh data/mmrag/index/
```

### Re-indexing a Document

If you need to re-index a document (e.g., after changing embedding model):

```bash
# Option 1: Delete manifest only (keeps images)
rm data/mmrag/docs/<doc_id>/manifest.json

# Option 2: Delete entire document
rm -rf data/mmrag/docs/<doc_id>

# Then re-upload via UI or API
```

### Clearing All Data

```bash
# Remove all documents and index
rm -rf data/mmrag/docs/*
rm -rf data/mmrag/index/*

# Or remove entire data directory
rm -rf data/mmrag
```

### Backing Up the Index

```bash
# Backup
cp -r data/mmrag/index data/mmrag/index.backup.$(date +%Y%m%d)

# Restore
rm -rf data/mmrag/index
cp -r data/mmrag/index.backup.20251215 data/mmrag/index
```

## Debugging Workflows

### Enabling Debug Logging

```python
# Add to app/api.py or run before starting
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via environment:
```bash
PYTHONUNBUFFERED=1 ./scripts/run_api.sh 2>&1 | tee api.log
```

### Inspecting Embeddings

```python
from app.rag.embedder import PageEmbedder
from pathlib import Path

embedder = PageEmbedder()

# Embed a single image
vec = embedder.embed_images([Path("data/mmrag/docs/<doc_id>/pages/page_0001.png")])
print(f"Shape: {vec.shape}")  # (1, 512)
print(f"Norm: {(vec**2).sum()**0.5}")  # Should be ~1.0

# Embed a query
qvec = embedder.embed_text("What is the revenue?")
print(f"Query shape: {qvec.shape}")  # (1, 512)
```

### Testing FAISS Search

```python
from app.rag.index_faiss import FaissPageIndex
from app.rag.config import settings

index = FaissPageIndex(
    index_path=settings.index_dir / "pages.faiss",
    meta_path=settings.index_dir / "pages.meta.json",
)
index.load()

print(f"Total pages indexed: {index.total_pages}")

# Inspect metadata
for ref in index.meta[:5]:
    print(f"  {ref.doc_id} page {ref.page_num}")
```

### Checking PDF Rendering

```python
from app.rag.pdf_pages import pdf_to_page_images
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    pages = pdf_to_page_images(
        Path("test.pdf"),
        Path(tmpdir),
        dpi=180
    )
    print(f"Rendered {len(pages)} pages")
    for p in pages[:3]:
        from PIL import Image
        img = Image.open(p)
        print(f"  {p.name}: {img.size}")
```

## CI/CD Workflows

### Running Tests (Future)

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Linting

```bash
# Format code
ruff format app/ ui/

# Check linting
ruff check app/ ui/

# Type checking
mypy app/ --ignore-missing-imports
```

### Building for Production

```bash
# Remove --reload from run_api.sh for production
# Use multiple workers:
uvicorn app.api:app --host 0.0.0.0 --port 3001 --workers 4
```

## Troubleshooting Workflows

### API Won't Start

```bash
# Check port availability
lsof -i :3001

# Kill existing process
pkill -f "uvicorn app.api"

# Check Python path
which python
python --version
```

### Model Download Issues

```bash
# Pre-download models
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"

# Check HuggingFace cache
ls ~/.cache/huggingface/hub/
```

### Memory Issues

```bash
# Monitor memory during ingestion
watch -n 1 free -h

# Check process memory
ps aux | grep python

# Reduce batch size if needed (edit embedder.py)
```

### Corrupted Index

```bash
# Symptoms: search returns wrong results or crashes

# Solution: Rebuild from scratch
rm -rf data/mmrag/index/*

# Remove all manifests to force re-indexing
find data/mmrag/docs -name "manifest.json" -delete

# Re-ingest all documents via UI
```
