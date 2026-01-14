# Code Reference

Technical documentation for each module in the system.

---

## app/rag/config.py

### Purpose
Centralized configuration management using Python dataclasses.

### Classes

#### `Settings`
```python
@dataclass(frozen=True)
class Settings:
    base_dir: Path = field(default_factory=_default_base)
    api_host: str = "0.0.0.0"
    api_port: int = 3001
    ui_host: str = "0.0.0.0"
    ui_port: int = 8081
```

**Properties:**
- `docs_dir` → `base_dir / "docs"`
- `index_dir` → `base_dir / "index"`
- `cache_dir` → `base_dir / "cache"`

**Design Notes:**
- `frozen=True` makes instances immutable (hashable, thread-safe)
- Properties ensure derived paths always reflect `base_dir`
- `field(default_factory=...)` is proper dataclass pattern for mutable defaults

### Module-level
```python
settings = Settings()  # Singleton instance
```

---

## app/rag/storage.py

### Purpose
File system operations and document state management.

### Functions

#### `ensure_dirs() -> None`
Creates all required storage directories if they don't exist.

#### `doc_id_from_bytes(data: bytes) -> str`
Generates a 16-character document ID from file content.
```python
# Uses SHA-256, truncated for readability
hashlib.sha256(data).hexdigest()[:16]
```

#### `doc_dir(doc_id: str) -> Path`
Returns: `settings.docs_dir / doc_id`

#### `pages_dir(doc_id: str) -> Path`
Returns: `doc_dir(doc_id) / "pages"`

#### `original_pdf_path(doc_id: str) -> Path`
Returns: `doc_dir(doc_id) / "original.pdf"`

#### `manifest_path(doc_id: str) -> Path`
Returns: `doc_dir(doc_id) / "manifest.json"`

#### `load_manifest(doc_id: str) -> DocManifest | None`
Loads and parses manifest JSON. Returns `None` if missing or corrupt.

#### `save_manifest(manifest: DocManifest) -> None`
Serializes and writes manifest to disk.

### Classes

#### `DocManifest`
```python
@dataclass
class DocManifest:
    doc_id: str
    filename: str = ""
    num_pages: int = 0
    indexed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    indexed_at: str | None = None
```

---

## app/rag/pdf_pages.py

### Purpose
Convert PDF documents to page images.

### Functions

#### `pdf_to_page_images(pdf_path, out_dir, dpi=180) -> list[Path]`

**Parameters:**
- `pdf_path: Path` - Input PDF file
- `out_dir: Path` - Output directory for images
- `dpi: int` - Rendering resolution (default 180)

**Returns:** List of paths to generated PNG files

**Implementation Details:**
```python
zoom = dpi / 72.0  # PDF base is 72 DPI
mat = fitz.Matrix(zoom, zoom)

with fitz.open(str(pdf_path)) as doc:
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img.save(out_dir / f"page_{i+1:04d}.png", optimize=True)
```

**Output Naming:** `page_0001.png`, `page_0002.png`, ... (1-indexed, zero-padded)

---

## app/rag/embedder.py

### Purpose
Generate vector embeddings for images and text using CLIP.

### Classes

#### `PageEmbedder`

**Constructor:**
```python
def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None)
```
- Auto-detects CUDA if available
- Loads model and processor from HuggingFace

**Methods:**

##### `embed_images(image_paths, batch_size=16) -> np.ndarray`
Embeds a list of images in batches.

**Returns:** `np.ndarray` of shape `(N, embed_dim)`, dtype `float32`, L2-normalized

##### `embed_text(query: str) -> np.ndarray`
Embeds a text query.

**Returns:** `np.ndarray` of shape `(1, embed_dim)`, dtype `float32`, L2-normalized

**Internal:**
```python
def _normalize(self, feats: torch.Tensor) -> np.ndarray:
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)
```

---

## app/rag/index_faiss.py

### Purpose
Vector index for efficient similarity search.

### Classes

#### `PageRef`
```python
@dataclass
class PageRef:
    doc_id: str
    page_num: int
    image_path: str
```

#### `FaissPageIndex`

**Constructor:**
```python
def __init__(self, index_path: Path, meta_path: Path)
```

**Methods:**

##### `load() -> None`
Loads FAISS index and metadata from disk.

##### `save() -> None`
Persists FAISS index and metadata to disk.

##### `add(vectors: np.ndarray, refs: list[PageRef]) -> None`
Adds vectors and their metadata to the index.

**Validation:**
- `vectors.dtype` must be `np.float32`
- `len(vectors)` must equal `len(refs)`

##### `search(query_vec, top_k=3) -> list[tuple[PageRef, float]]`
Finds the most similar vectors.

**Returns:** List of `(PageRef, score)` tuples, sorted by descending similarity

**Properties:**
- `total_pages: int` - Number of vectors in the index

**Index Type:** `faiss.IndexFlatIP` (Inner Product = cosine similarity on normalized vectors)

---

## app/rag/vlm_qwen25vl.py

### Purpose
Vision-language model for answering questions based on images.

### Classes

#### `VLMAnswerer`

**Constructor:**
```python
def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct")
```
*Currently a stub - model not loaded*

**Methods:**

##### `answer(question: str, evidence_images: list[Path]) -> str`
Generates an answer based on question and evidence images.

**Current Implementation:** Returns stub response with metadata.

**TODO:** Implement actual Qwen2.5-VL inference:
```python
# Pseudocode for future implementation
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": img} for img in images
    ] + [{"type": "text", "text": question}]}
]
response = model.generate(messages)
```

---

## app/rag/pipeline.py

### Purpose
Main orchestrator connecting all components.

### Classes

#### `MMRagPipeline`

**Constructor:**
```python
def __init__(self):
    ensure_dirs()
    self.embedder = PageEmbedder()
    self.answerer = VLMAnswerer()
    self.index = FaissPageIndex(...)
    self.index.load()
```

**Methods:**

##### `ingest_pdf_bytes(pdf_bytes, filename="upload.pdf") -> dict`

**Flow:**
1. Generate `doc_id` from content hash
2. Check manifest for existing index
3. If indexed, return cached result
4. Save PDF to disk
5. Convert to page images
6. Embed all pages
7. Add to FAISS index
8. Save manifest with `indexed=True`

**Returns:**
```python
{
    "doc_id": str,
    "num_pages": int,
    "is_new": bool
}
```

##### `chat(question: str, top_k: int = 3) -> dict`

**Flow:**
1. Check if index is empty
2. Embed question text
3. Search FAISS for top-K pages
4. Load evidence page images
5. Generate answer with VLM

**Returns:**
```python
{
    "answer": str,
    "evidence": [
        {"doc_id": str, "page_num": int, "image_path": str, "score": float},
        ...
    ]
}
```

---

## app/api.py

### Purpose
REST API gateway using FastAPI.

### Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    pipe = MMRagPipeline()
    yield
```

Pipeline is initialized once at startup, not on import.

### Endpoints

| Method | Path | Function | Request | Response |
|--------|------|----------|---------|----------|
| GET | `/health` | `health()` | - | `dict` |
| POST | `/ingest` | `ingest(file)` | `UploadFile` | `IngestResponse` |
| POST | `/chat` | `chat(req)` | `ChatRequest` | `ChatResponse` |

### Models

```python
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

class IngestResponse(BaseModel):
    doc_id: str
    num_pages: int
    is_new: bool

class ChatResponse(BaseModel):
    answer: str
    evidence: list[dict]
```

---

## ui/gradio_app.py

### Purpose
Web interface using Gradio Blocks.

### Functions

#### `check_api_health() -> str`
Called on page load to verify API connection.

#### `ingest_pdf(pdf_file) -> str`
Uploads PDF to API and returns status message.

#### `chat(question, top_k) -> tuple[str, list[str]]`
Sends question to API, returns `(answer, evidence_paths)`.

### UI Components

```python
with gr.Blocks() as demo:
    # Status
    api_status = gr.Textbox()
    
    # Ingest section
    pdf = gr.File(file_types=[".pdf"])
    ingest_btn = gr.Button("Ingest")
    ingest_status = gr.Textbox()
    
    # Chat section
    q = gr.Textbox(label="Question")
    top_k = gr.Slider(1, 8, value=3)
    ask_btn = gr.Button("Ask")
    answer_box = gr.Textbox()
    gallery = gr.Gallery()
```

### Configuration

Environment variables:
- `API_BASE` - API URL (default: `http://127.0.0.1:3001`)
- `UI_HOST` - Bind address (default: `0.0.0.0`)
- `UI_PORT` - Port (default: `8081`)
