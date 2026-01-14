"""Vision-Language Model answering using Qwen2.5-VL."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VLMAnswerer:
    """
    VLM-based question answering over document page images using Qwen2.5-VL.
    
    Implements lazy model loading with graceful fallback to stub responses.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the VLM answerer.

        Args:
            model_name: HuggingFace model ID for Qwen2.5-VL.
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self._model_load_failed = False

    def _load_model(self):
        """
        Lazily load the Qwen2.5-VL model and processor.
        
        Uses GPU if available, otherwise falls back to CPU.
        Sets _model_load_failed flag if loading fails.
        """
        if self.model is not None or self._model_load_failed:
            return
        
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading {self.model_name} on device: {self.device.upper()}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            self.model.eval()
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load VLM model: {e}. Falling back to stub responses.")
            self._model_load_failed = True
            self.model = None
            self.processor = None

    def _validate_images(self, image_paths: list[Path]) -> list[Image.Image]:
        """Load and validate that all evidence images are readable."""
        images = []
        for p in image_paths:
            if not p.exists():
                raise FileNotFoundError(f"Evidence image not found: {p}")
            images.append(Image.open(p).convert("RGB"))
        return images

    def _generate_with_vlm(self, question: str, images: list[Image.Image]) -> str:
        """
        Generate answer using Qwen2.5-VL model.
        
        Args:
            question: User's question.
            images: List of PIL images (evidence pages).
            
        Returns:
            Generated answer string.
        """
        try:
            # Build multimodal prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based strictly on the provided document pages. If the answer is not present in the pages, explicitly state that you cannot find the information."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the following {len(images)} document page(s), please answer this question: {question}\n\nProvide a clear answer and indicate which pages you used."
                        }
                    ] + [
                        {
                            "type": "image",
                            "image": img
                        } for img in images
                    ]
                }
            ]
            
            # Prepare inputs
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_prompt],
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Reduced for faster CPU inference
                    do_sample=False
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            # Extract answer (remove prompt)
            # The generated text includes the full conversation, extract only the assistant's response
            if "assistant\n" in generated_text:
                answer = generated_text.split("assistant\n")[-1].strip()
            else:
                answer = generated_text.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            raise

    def _format_with_citations(self, answer: str, num_pages: int) -> str:
        """
        Add citation footer to the answer.
        
        Args:
            answer: Generated answer text.
            num_pages: Number of evidence pages used.
            
        Returns:
            Answer with citation footer.
        """
        page_refs = ", ".join([f"p.{i+1}" for i in range(num_pages)])
        return f"{answer}\n\nEvidence pages: {page_refs}"

    def _stub_response(self, question: str, num_images: int) -> str:
        """
        Generate stub response when model is unavailable.
        
        Args:
            question: User's question.
            num_images: Number of evidence pages.
            
        Returns:
            Stub response string.
        """
        page_refs = ", ".join([f"p.{i+1}" for i in range(num_images)])
        return "\n".join([
            "[VLM Stub] Model not loaded. This is a placeholder response.",
            f"Question: {question}",
            "",
            "The system would analyze the provided document pages to answer your question.",
            f"\nEvidence pages: {page_refs}",
        ])

    def answer(self, question: str, evidence_images: list[Path]) -> str:
        """
        Generate an answer based on the question and evidence page images.

        Args:
            question: The user's question.
            evidence_images: Paths to retrieved page images.

        Returns:
            Generated answer string with citations.
        """
        # Validate images
        images = self._validate_images(evidence_images)
        
        # Lazy load model
        self._load_model()
        
        # Generate answer or fall back to stub
        if self.model is not None and self.processor is not None:
            try:
                answer = self._generate_with_vlm(question, images)
                return self._format_with_citations(answer, len(images))
            except Exception as e:
                logger.error(f"VLM inference failed, falling back to stub: {e}")
                return self._stub_response(question, len(images))
        else:
            return self._stub_response(question, len(images))
