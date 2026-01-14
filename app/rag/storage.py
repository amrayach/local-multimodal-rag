"""Storage utilities for document management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import settings


def ensure_dirs() -> None:
    """Create all required storage directories."""
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)


def doc_id_from_bytes(data: bytes) -> str:
    """Generate a unique document ID from file contents."""
    return hashlib.sha256(data).hexdigest()[:16]


def sha256_from_bytes(data: bytes) -> str:
    """Generate full SHA256 hash from file contents."""
    return hashlib.sha256(data).hexdigest()


def doc_dir(doc_id: str) -> Path:
    """Get the root directory for a document."""
    return settings.docs_dir / doc_id


def pages_dir(doc_id: str) -> Path:
    """Get the pages directory for a document."""
    return doc_dir(doc_id) / "pages"


def original_pdf_path(doc_id: str) -> Path:
    """Get the path to the original PDF file."""
    return doc_dir(doc_id) / "original.pdf"


def manifest_path(doc_id: str) -> Path:
    """Get the path to the document manifest file."""
    return doc_dir(doc_id) / "manifest.json"


@dataclass
class DocManifest:
    """Manifest tracking document ingestion state."""

    doc_id: str
    filename: str = ""
    num_pages: int = 0
    indexed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    indexed_at: str | None = None
    # New fields for idempotent ingest
    sha256: str = ""
    index_backend: str = "faiss"
    embedder_id: str = ""


def load_manifest(doc_id: str) -> DocManifest | None:
    """Load manifest for a document, or None if not found."""
    path = manifest_path(doc_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DocManifest(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def save_manifest(manifest: DocManifest) -> None:
    """Save manifest to disk."""
    path = manifest_path(manifest.doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
