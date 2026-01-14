"""PDF to image conversion utilities."""

from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

DPI_DEFAULT = 180
PDF_BASE_DPI = 72.0


def pdf_to_page_images(pdf_path: Path, out_dir: Path, dpi: int = DPI_DEFAULT) -> list[Path]:
    """
    Convert each page of a PDF to a PNG image.

    Args:
        pdf_path: Path to the input PDF file.
        out_dir: Directory to save the page images.
        dpi: Resolution for rendering (default 180).

    Returns:
        List of paths to the generated PNG images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / PDF_BASE_DPI
    mat = fitz.Matrix(zoom, zoom)
    out_paths: list[Path] = []

    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            p = out_dir / f"page_{i + 1:04d}.png"
            img.save(p, optimize=True)
            out_paths.append(p)

    return out_paths
