"""FAISS-based vector index for page retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import faiss
import numpy as np


@dataclass
class PageRef:
    """Reference to a specific page in a document."""

    doc_id: str
    page_num: int
    image_path: str


class FaissPageIndex:
    """
    Simple cosine-similarity index using inner product on normalized vectors.
    """

    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: faiss.IndexFlatIP | None = None
        self.meta: list[PageRef] = []

    def _save_meta(self) -> None:
        """Persist page metadata to JSON."""
        payload = [asdict(ref) for ref in self.meta]
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_meta(self) -> None:
        """Load page metadata from JSON."""
        if self.meta_path.exists():
            payload = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.meta = [PageRef(**d) for d in payload]
        else:
            self.meta = []

    def load(self) -> None:
        """Load index and metadata from disk."""
        self._load_meta()
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self.index is not None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
        self._save_meta()

    def add(self, vectors: np.ndarray, refs: list[PageRef]) -> None:
        """
        Add page vectors and their references to the index.

        Args:
            vectors: Normalized embeddings of shape (N, dim), dtype float32.
            refs: Corresponding page references.
        """
        if vectors.dtype != np.float32:
            raise ValueError(f"Expected float32 vectors, got {vectors.dtype}")
        if len(vectors) != len(refs):
            raise ValueError(f"Vector count ({len(vectors)}) != ref count ({len(refs)})")

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vectors)
        self.meta.extend(refs)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[tuple[PageRef, float]]:
        """
        Search for the most similar pages.

        Args:
            query_vec: Normalized query embedding of shape (1, dim).
            top_k: Number of results to return.

        Returns:
            List of (PageRef, score) tuples, sorted by descending similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Ingest a document first.")

        scores, idxs = self.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(idxs[0], scores[0]):
            if 0 <= idx < len(self.meta):
                results.append((self.meta[idx], float(score)))

        return results

    @property
    def total_pages(self) -> int:
        """Return the total number of indexed pages."""
        return self.index.ntotal if self.index else 0
