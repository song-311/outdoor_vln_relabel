"""VLM client abstraction layer with Anthropic Claude and local-vLLM backends."""

from __future__ import annotations

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: str) -> Dict[str, Any]:
    """Read a local image file and return a Claude-compatible base64 content block.

    Args:
        image_path: Path to a .png, .jpg, or .jpeg file.

    Returns:
        Dict with ``type``, ``source.media_type``, and ``source.data`` keys.
    """
    ext = os.path.splitext(image_path)[1].lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_map.get(ext, "image/png")

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


class BaseVLMClient(ABC):
    """Abstract interface for VLM backends.

    Subclass and implement :meth:`generate` to support a new backend.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: List[str],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        """Send a prompt with images to the VLM and return the raw text response.

        Args:
            prompt: The formatted prompt string.
            images: List of local image file paths.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            **kwargs: Backend-specific extra parameters.

        Returns:
            Raw text response from the model.
        """
        raise NotImplementedError


class ClaudeVLMClient(BaseVLMClient):
    """VLM client backed by the Anthropic Claude API (Messages endpoint).

    Reads the API key from the ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key= to ClaudeVLMClient."
            )

    def generate(
        self,
        prompt: str,
        images: List[str],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for ClaudeVLMClient. Install with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self._api_key)

        # Build content blocks: images first, then text prompt
        content_blocks: List[Dict[str, Any]] = []
        for img_path in images:
            content_blocks.append(encode_image_base64(img_path))
        content_blocks.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content_blocks}],
            **kwargs,
        )

        # Extract text from the first text block
        for block in response.content:
            if block.type == "text":
                return block.text

        logger.warning("Claude response contained no text block. Full response: %s", response)
        return ""

    def generate_structured(
        self,
        prompt: str,
        images: List[str],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call :meth:`generate` and parse the result as JSON.

        Returns:
            Parsed JSON dict, or a dict with an ``_error`` key on failure.
        """
        raw = self.generate(
            prompt, images, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        return _parse_vlm_json(raw)


class LocalVLLMClient(BaseVLMClient):
    """Placeholder for an OpenAI-compatible local vLLM / Ollama endpoint.

    Usage (future)::

        client = LocalVLLMClient(base_url="http://localhost:8000/v1", model="Qwen2.5-VL-7B")
        text = client.generate(prompt, images)
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "Qwen2.5-VL-7B") -> None:
        self.base_url = base_url
        self.model = model

    def generate(
        self,
        prompt: str,
        images: List[str],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            "LocalVLLMClient is not yet implemented. "
            "Install the openai package and use an OpenAI-compatible endpoint."
        )


def _parse_vlm_json(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from VLM text output.

    Handles common issues like markdown code fences and trailing commas.
    """
    if not raw or not raw.strip():
        return {"_error": "empty_response"}

    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening ```json or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse VLM response as JSON: %s", raw[:500])
    return {"_error": "json_parse_failed", "_raw": raw}
