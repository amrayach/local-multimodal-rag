"""Visual page embedding using CLIP."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class PageEmbedder:
    """
    Baseline visual page embedder using CLIP.
    
    TODO: Replace with ColPali/ColQwen2 for better document understanding.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _normalize(self, feats: torch.Tensor) -> np.ndarray:
        """L2-normalize and convert to numpy float32."""
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_images(self, image_paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """
        Embed a list of page images.

        Args:
            image_paths: Paths to PNG/JPG images.
            batch_size: Number of images to process at once.

        Returns:
            Normalized embeddings of shape (N, embed_dim).
        """
        all_feats = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            all_feats.append(self._normalize(feats))

        return np.vstack(all_feats)

    @torch.inference_mode()
    def embed_text(self, query: str) -> np.ndarray:
        """
        Embed a text query.

        Args:
            query: The search query string.

        Returns:
            Normalized embedding of shape (1, embed_dim).
        """
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return self._normalize(feats)
