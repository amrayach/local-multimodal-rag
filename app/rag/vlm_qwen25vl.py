"""Vision-Language Model answering using Qwen2.5-VL."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class VLMAnswerer:
    """
    VLM-based question answering over document page images.

    TODO: Replace stub with Qwen2.5-VL inference once model loading is validated.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the VLM answerer.

        Args:
            model_name: HuggingFace model ID (unused in stub).
        """
        self.model_name = model_name
        self.model = None  # Lazy load when implemented

    def _validate_images(self, image_paths: list[Path]) -> list[Image.Image]:
        """Load and validate that all evidence images are readable."""
        images = []
        for p in image_paths:
            if not p.exists():
                raise FileNotFoundError(f"Evidence image not found: {p}")
            images.append(Image.open(p).convert("RGB"))
        return images

    def answer(self, question: str, evidence_images: list[Path]) -> str:
        """
        Generate an answer based on the question and evidence page images.

        Args:
            question: The user's question.
            evidence_images: Paths to retrieved page images.

        Returns:
            Generated answer string.
        """
        images = self._validate_images(evidence_images)

        # --- Stub response (replace with actual Qwen2.5-VL inference) ---
        return "\n".join([
            "[VLM Stub] This is a placeholder response.",
            f"Question: {question}",
            f"Evidence pages: {len(images)}",
            "",
            "TODO: Integrate Qwen2.5-VL to generate grounded answers from page images.",
        ])
