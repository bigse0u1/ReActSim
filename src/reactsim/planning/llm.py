"""Thin client for the Meta Model API (Muse Spark).

Uses the OpenAI-compatible Chat Completions endpoint (model e.g. "muse-spark-1.3").
See https://dev.meta.ai/docs/quickstart.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class LLMCallLog:
    latency_s: float
    prompt_chars: int
    response_chars: int
    usage: dict = field(default_factory=dict)


class MetaChatClient:
    def __init__(self, model: str, base_url: str | None = None,
                 temperature: float = 0.2, timeout: float = 60.0):
        import httpx

        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1")
        self.api_key = os.environ["MODEL_API_KEY"]   # never log this
        self.client = httpx.Client(timeout=timeout)
        self.calls: list[LLMCallLog] = []

    def chat(self, system: str, user: str, images: list | None = None) -> str:
        """images: optional RGB uint8 arrays sent with the user message (Muse Spark is multimodal)."""
        t0 = time.time()
        content = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {"type": "image_url", "image_url": {"url": _jpeg_data_url(im)}} for im in images]
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            },
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Muse request failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        self.calls.append(LLMCallLog(time.time() - t0, len(system) + len(user),
                                     len(text), data.get("usage", {})))
        return text


def _jpeg_data_url(rgb) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
