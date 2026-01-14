# API Reference

## Base URL

```
http://localhost:3001
```

## Authentication

Currently no authentication is required. All endpoints are open.

---

## Endpoints

### GET /health

Health check endpoint for monitoring and load balancers.

#### Request

```http
GET /health HTTP/1.1
Host: localhost:3001
```

#### Response

```json
{
  "ok": true,
  "indexed_pages": 142
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Always `true` if server is running |
| `indexed_pages` | integer | Total number of pages in the FAISS index |

#### Example

```bash
curl http://localhost:3001/health
```

---

### GET /stats

System statistics endpoint for observability and monitoring.

#### Request

```http
GET /stats HTTP/1.1
Host: localhost:3001
```

#### Response

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

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `num_docs` | integer | Number of indexed documents |
| `num_indexed_pages` | integer | Total pages in the index |
| `embed_dim` | integer | Embedding vector dimension |
| `embedder_id` | string | Model identifier for embeddings |
| `index_backend` | string | Vector index backend (faiss) |
| `index_type` | string | FAISS index type (IndexFlatIP) |
| `device` | string | Compute device (cpu/cuda) |
| `gpu_name` | string \| null | GPU name if CUDA available |
| `gpu_memory_mb` | float \| null | GPU memory allocated (MB) |
| `process_memory_mb` | float \| null | Process memory usage (MB) |
| `demo_limits` | object | Configured demo limits |

#### Example

```bash
curl http://localhost:3001/stats | jq .
```

---

### POST /ingest

Upload and index a PDF document.

#### Request

```http
POST /ingest HTTP/1.1
Host: localhost:3001
Content-Type: multipart/form-data

file: <binary PDF data>
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | PDF file to ingest (multipart form) |

#### Response

```json
{
  "doc_id": "a1b2c3d4e5f67890",
  "num_pages": 35,
  "is_new": true
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | 16-character hex ID (SHA-256 prefix of content) |
| `num_pages` | integer | Number of pages in the document |
| `is_new` | boolean | `true` if newly indexed, `false` if already existed |

#### Error Responses

| Status | Condition | Response |
|--------|-----------|----------|
| 400 | Non-PDF file | `{"detail": "Only PDF files are supported. Please upload a .pdf file."}` |
| 400 | Empty file | `{"detail": "Empty file uploaded. Please select a valid PDF."}` |
| 400 | File too large | `{"detail": "File too large: 55.2MB exceeds 50MB limit"}` |
| 400 | Too many pages | `{"detail": "Too many pages: 150 exceeds 100 page limit"}` |
| 500 | Processing error | `{"detail": "Ingest failed: <error message>"}` |

#### Demo Limits

| Limit | Value | Description |
|-------|-------|-------------|
| Max file size | 50 MB | Maximum PDF file size |
| Max pages | 100 | Maximum pages per document |
| Max DPI | 180 | Maximum rendering resolution |

#### Example

```bash
# Upload a PDF
curl -X POST http://localhost:3001/ingest \
  -F "file=@annual_report.pdf"

# Response
{
  "doc_id": "7f83b1657ff1fc53",
  "num_pages": 42,
  "is_new": true
}

# Re-uploading same file
curl -X POST http://localhost:3001/ingest \
  -F "file=@annual_report.pdf"

# Response (idempotent)
{
  "doc_id": "7f83b1657ff1fc53",
  "num_pages": 42,
  "is_new": false
}
```

---

### POST /chat

Ask a question about ingested documents.

#### Request

```http
POST /chat HTTP/1.1
Host: localhost:3001
Content-Type: application/json

{
  "question": "What are the key findings?",
  "top_k": 3
}
```

#### Request Body

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `question` | string | Yes | - | min 1 char | The question to answer |
| `top_k` | integer | No | 3 | 1-10 | Number of evidence pages to retrieve |

#### Response

```json
{
  "answer": "Based on the document, the key findings are...",
  "evidence": [
    {
      "doc_id": "7f83b1657ff1fc53",
      "page_num": 12,
      "image_path": "/home/ammer/local_multimodal_rag/data/mmrag/docs/7f83b1657ff1fc53/pages/page_0012.png",
      "score": 0.8542
    },
    {
      "doc_id": "7f83b1657ff1fc53",
      "page_num": 5,
      "image_path": "/home/ammer/local_multimodal_rag/data/mmrag/docs/7f83b1657ff1fc53/pages/page_0005.png",
      "score": 0.7891
    }
  ]
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | Generated answer from the VLM |
| `evidence` | array | List of retrieved evidence pages |
| `evidence[].doc_id` | string | Document ID containing this page |
| `evidence[].page_num` | integer | 1-indexed page number |
| `evidence[].image_path` | string | Absolute path to page image |
| `evidence[].score` | float | Cosine similarity score (0-1) |

#### Error Responses

| Status | Condition | Response |
|--------|-----------|----------|
| 422 | Missing question | `{"detail": [{"loc": ["body", "question"], ...}]}` |
| 422 | Invalid top_k | `{"detail": [{"loc": ["body", "top_k"], ...}]}` |

#### Special Cases

**Empty Index:**
```json
{
  "answer": "No documents indexed yet. Please upload a PDF first.",
  "evidence": []
}
```

#### Example

```bash
# Ask a question
curl -X POST http://localhost:3001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was the total revenue in 2024?",
    "top_k": 5
  }'
```

---

## Data Types

### ChatRequest

```python
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
```

### IngestResponse

```python
class IngestResponse(BaseModel):
    doc_id: str
    num_pages: int
    is_new: bool
```

### ChatResponse

```python
class ChatResponse(BaseModel):
    answer: str
    evidence: list[dict]
```

---

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:3001/docs
- **ReDoc**: http://localhost:3001/redoc
- **OpenAPI JSON**: http://localhost:3001/openapi.json

---

## Rate Limiting

No rate limiting is currently implemented. For production deployments, consider adding:

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
def chat(req: ChatRequest):
    ...
```

---

## CORS

CORS is not configured by default. To enable for web frontends:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
