"""Main RAG pipeline orchestrating ingestion and retrieval."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import settings
from .embedder import PageEmbedder, DEFAULT_MODEL
from .index_faiss import FaissPageIndex, PageRef
from .pdf_pages import pdf_to_page_images
from .storage import (
    DocManifest,
    doc_dir,
    doc_id_from_bytes,
    sha256_from_bytes,
    ensure_dirs,
    load_manifest,
    original_pdf_path,
    pages_dir,
    save_manifest,
)
from .vlm_qwen25vl import VLMAnswerer

logger = logging.getLogger(__name__)

# Demo limits
MAX_PAGES = 100
MAX_FILE_SIZE_MB = 50
MAX_DPI = 180


class IngestError(Exception):
    """Raised when document ingestion fails due to validation or limits."""
    pass


class MMRagPipeline:
    """
    Multimodal RAG pipeline for PDF document Q&A.

    Handles PDF ingestion, page embedding, indexing, and VLM-based answering.
    """

    def __init__(self):
        ensure_dirs()
        self.embedder = PageEmbedder()
        self.answerer = VLMAnswerer()
        self.index = FaissPageIndex(
            index_path=settings.index_dir / "pages.faiss",
            meta_path=settings.index_dir / "pages.meta.json",
        )
        self.index.load()
        logger.info(f"Pipeline initialized with {self.index.total_pages} indexed pages")

    @property
    def embedder_id(self) -> str:
        """Return identifier for the current embedder model."""
        return DEFAULT_MODEL

    def ingest_pdf_bytes(self, pdf_bytes: bytes, filename: str = "upload.pdf") -> dict:
        """
        Ingest a PDF document: convert to images, embed, and index.

        Args:
            pdf_bytes: Raw PDF file content.
            filename: Original filename (for logging).

        Returns:
            Dict with doc_id, num_pages, and is_new flag.

        Raises:
            IngestError: If file exceeds limits or is invalid.
        """
        from datetime import datetime
        import fitz

        total_start = time.perf_counter()
        timings = {}

        # Validate file size
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise IngestError(f"File too large: {file_size_mb:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit")

        # Compute identifiers
        sha256 = sha256_from_bytes(pdf_bytes)
        doc_id = doc_id_from_bytes(pdf_bytes)
        ddir = doc_dir(doc_id)
        ddir.mkdir(parents=True, exist_ok=True)

        # Check existing manifest for idempotent ingest
        manifest = load_manifest(doc_id)
        if manifest is not None and manifest.indexed:
            # Already indexed - skip re-adding to FAISS
            logger.info(
                f"Document already indexed (idempotent skip): doc_id={doc_id}, "
                f"sha256={manifest.sha256[:16]}..., pages={manifest.num_pages}"
            )
            return {
                "doc_id": doc_id,
                "num_pages": manifest.num_pages,
                "is_new": False,
            }

        # Save original PDF if not already present
        pdf_path = original_pdf_path(doc_id)
        if not pdf_path.exists():
            pdf_path.write_bytes(pdf_bytes)
            logger.info(f"Saved new PDF: {filename} -> {doc_id}")

        # Validate page count before rendering
        with fitz.open(str(pdf_path)) as doc:
            num_pages = len(doc)
            if num_pages > MAX_PAGES:
                raise IngestError(f"Too many pages: {num_pages} exceeds {MAX_PAGES} page limit")

        # Convert to page images (skip if already done)
        pdir = pages_dir(doc_id)
        page_imgs = sorted(pdir.glob("page_*.png"))
        if not page_imgs:
            logger.info(f"Converting PDF to page images: {doc_id}")
            t0 = time.perf_counter()
            page_imgs = pdf_to_page_images(pdf_path, pdir, dpi=min(MAX_DPI, 180))
            timings["pdf_render_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Embed pages
        t0 = time.perf_counter()
        vecs = self.embedder.embed_images(page_imgs)
        timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Add to index
        t0 = time.perf_counter()
        refs = [
            PageRef(doc_id=doc_id, page_num=i, image_path=str(img_path))
            for i, img_path in enumerate(page_imgs, start=1)
        ]
        self.index.add(vecs, refs)
        self.index.save()
        timings["index_add_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 1)

        # Save manifest to mark as indexed
        manifest = DocManifest(
            doc_id=doc_id,
            filename=filename,
            num_pages=len(page_imgs),
            indexed=True,
            indexed_at=datetime.now().isoformat(),
            sha256=sha256,
            index_backend="faiss",
            embedder_id=self.embedder_id,
        )
        save_manifest(manifest)

        logger.info(
            f"Indexed {len(page_imgs)} pages for doc {doc_id} | "
            f"timings: render={timings.get('pdf_render_ms', 'cached')}ms, "
            f"embed={timings['embed_ms']}ms, index={timings['index_add_ms']}ms, "
            f"total={timings['total_ms']}ms"
        )

        return {
            "doc_id": doc_id,
            "num_pages": len(page_imgs),
            "is_new": True,
        }

    def chat(self, question: str, top_k: int = 3) -> dict:
        """
        Answer a question using retrieved page evidence.

        Args:
            question: User's question.
            top_k: Number of pages to retrieve.

        Returns:
            Dict with answer and evidence list.
        """
        if self.index.total_pages == 0:
            return {
                "answer": "No documents indexed yet. Please upload a PDF first.",
                "evidence": [],
            }

        qv = self.embedder.embed_text(question)
        hits = self.index.search(qv, top_k=top_k)

        evidence = [
            {
                "doc_id": ref.doc_id,
                "page_num": ref.page_num,
                "image_path": ref.image_path,
                "score": round(score, 4),
            }
            for ref, score in hits
        ]

        evidence_paths = [Path(e["image_path"]) for e in evidence]
        answer = self.answerer.answer(question, evidence_paths)

        logger.debug(f"Query: {question[:50]}... -> {len(evidence)} hits")

        return {"answer": answer, "evidence": evidence}

    def get_stats(self) -> dict:
        """
        Return statistics about the pipeline for observability.

        Returns:
            Dict with document/page counts, embedding info, device info, memory usage.
        """
        import torch
        import os

        # Count documents
        docs_dir = settings.docs_dir
        doc_ids = []
        total_doc_pages = 0
        if docs_dir.exists():
            for d in docs_dir.iterdir():
                if d.is_dir():
                    manifest = load_manifest(d.name)
                    if manifest and manifest.indexed:
                        doc_ids.append(d.name)
                        total_doc_pages += manifest.num_pages

        # Device info
        device = self.embedder.device
        gpu_name = None
        gpu_memory_mb = None
        if device == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)

        # Embedding dimension
        embed_dim = self.index.index.d if self.index.index else 512

        # Process memory (best-effort)
        try:
            import resource
            mem_usage_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            mem_usage_mb = None

        return {
            "num_docs": len(doc_ids),
            "num_indexed_pages": self.index.total_pages,
            "embed_dim": embed_dim,
            "embedder_id": self.embedder_id,
            "index_backend": "faiss",
            "index_type": "IndexFlatIP",
            "device": device,
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory_mb,
            "process_memory_mb": mem_usage_mb,
            "demo_limits": {
                "max_pages": MAX_PAGES,
                "max_file_size_mb": MAX_FILE_SIZE_MB,
                "max_dpi": MAX_DPI,
            },
        }

    def clear_index(self) -> None:
        """Clear the FAISS index and metadata (but keep document files)."""
        import shutil

        # Remove index files
        if settings.index_dir.exists():
            shutil.rmtree(settings.index_dir)
        settings.index_dir.mkdir(parents=True, exist_ok=True)

        # Reset in-memory index
        self.index = FaissPageIndex(
            index_path=settings.index_dir / "pages.faiss",
            meta_path=settings.index_dir / "pages.meta.json",
        )

        # Mark all manifests as not indexed
        if settings.docs_dir.exists():
            for d in settings.docs_dir.iterdir():
                if d.is_dir():
                    manifest = load_manifest(d.name)
                    if manifest:
                        manifest.indexed = False
                        manifest.indexed_at = None
                        save_manifest(manifest)

        logger.info("Index cleared")

    def reindex_all(self) -> dict:
        """
        Rebuild the index from all existing documents.

        Returns:
            Dict with counts of reindexed documents and pages.
        """
        from datetime import datetime
        import time

        total_start = time.perf_counter()
        docs_reindexed = 0
        pages_reindexed = 0

        if not settings.docs_dir.exists():
            return {"docs": 0, "pages": 0, "elapsed_ms": 0}

        for d in sorted(settings.docs_dir.iterdir()):
            if not d.is_dir():
                continue

            doc_id = d.name
            pdf_path = original_pdf_path(doc_id)
            pdir = pages_dir(doc_id)

            if not pdf_path.exists():
                logger.warning(f"Skipping {doc_id}: no original.pdf found")
                continue

            # Get page images
            page_imgs = sorted(pdir.glob("page_*.png"))
            if not page_imgs:
                logger.info(f"Rendering pages for {doc_id}")
                page_imgs = pdf_to_page_images(pdf_path, pdir, dpi=min(MAX_DPI, 180))

            if not page_imgs:
                logger.warning(f"Skipping {doc_id}: no pages found")
                continue

            # Embed and add to index
            vecs = self.embedder.embed_images(page_imgs)
            refs = [
                PageRef(doc_id=doc_id, page_num=i, image_path=str(img_path))
                for i, img_path in enumerate(page_imgs, start=1)
            ]
            self.index.add(vecs, refs)

            # Update manifest
            manifest = load_manifest(doc_id) or DocManifest(doc_id=doc_id)
            manifest.indexed = True
            manifest.indexed_at = datetime.now().isoformat()
            manifest.num_pages = len(page_imgs)
            manifest.sha256 = sha256_from_bytes(pdf_path.read_bytes())
            manifest.index_backend = "faiss"
            manifest.embedder_id = self.embedder_id
            save_manifest(manifest)

            docs_reindexed += 1
            pages_reindexed += len(page_imgs)
            logger.info(f"Reindexed {doc_id}: {len(page_imgs)} pages")

        self.index.save()
        elapsed_ms = round((time.perf_counter() - total_start) * 1000, 1)

        logger.info(
            f"Reindex complete: {docs_reindexed} docs, {pages_reindexed} pages, {elapsed_ms}ms"
        )

        return {
            "docs": docs_reindexed,
            "pages": pages_reindexed,
            "elapsed_ms": elapsed_ms,
        }
