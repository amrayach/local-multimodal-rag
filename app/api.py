"""FastAPI gateway for the Multimodal RAG service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.pipeline import MMRagPipeline, IngestError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

pipe: MMRagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipe
    pipe = MMRagPipeline()
    yield
    # Cleanup if needed


app = FastAPI(
    title="Local MMRAG Gateway",
    description="Multimodal RAG API for PDF document Q&A",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of pages to retrieve")


class IngestResponse(BaseModel):
    """Response from document ingestion."""

    doc_id: str
    num_pages: int
    is_new: bool


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    answer: str
    evidence: list[dict]


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"ok": True, "indexed_pages": pipe.index.total_pages if pipe else 0}


@app.get("/stats")
def stats() -> dict:
    """
    Return system statistics for observability.

    Includes document/page counts, embedding info, device info, memory usage.
    """
    if not pipe:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipe.get_stats()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(..., description="PDF file to ingest")):
    """
    Ingest a PDF document into the RAG index.

    The PDF is converted to page images, embedded, and indexed for retrieval.
    Demo limits: max 100 pages, max 50MB file size.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported. Please upload a .pdf file.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded. Please select a valid PDF.")

    try:
        result = pipe.ingest_pdf_bytes(pdf_bytes, filename=file.filename)
        logger.info(f"Ingest completed: doc_id={result['doc_id']}, pages={result['num_pages']}, is_new={result['is_new']}")
        return result
    except IngestError as e:
        # Graceful error for demo limits
        logger.warning(f"Ingest rejected: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during ingest: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Ask a question about the ingested documents.

    Returns an answer generated from the most relevant pages.
    """
    try:
        return pipe.chat(req.question, top_k=req.top_k)
    except RuntimeError as e:
        # Handle empty index gracefully
        logger.warning(f"Chat failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
