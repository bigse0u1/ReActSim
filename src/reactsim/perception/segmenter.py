"""Text-prompted segmentation: SAM 3.1 (Meta API) and a simulator oracle.

The oracle uses ManiSkill's ground-truth segmentation. It is the upper bound
SAM 3.1 is evaluated against, and the fallback while the API is being wired up.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Mask:
    prompt: str
    mask: np.ndarray        # HxW bool
    score: float


class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, Mask]: ...


class OracleSegmenter:
    """prompt ("banana") -> actor name ("banana") -> GT segmentation id."""

    def __init__(self, seg_ids: dict[str, int]):
        self.seg_ids = seg_ids
        self.segmentation: np.ndarray | None = None   # set per frame by the perception module

    def segment(self, rgb, prompts):
        out = {}
        for p in prompts:
            name = p.strip().lower().replace(" ", "_")
            if name in self.seg_ids and self.segmentation is not None:
                m = self.segmentation == self.seg_ids[name]
                if m.any():
                    out[p] = Mask(p, m, 1.0)
        return out


class Sam3Segmenter:
    """SAM 3.1 via the Meta Model API.

    TODO(day 3): the request/response schema below is a placeholder. Check
    https://dev.meta.ai/docs/models for the real SAM 3.1 endpoint, input format
    (image + text concept prompt) and mask encoding (RLE vs PNG vs polygons).
    """

    def __init__(self, model: str = "sam-3.1", base_url: str | None = None, timeout: float = 30.0):
        import httpx

        self.model = model
        self.base_url = base_url or os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1")
        self.client = httpx.Client(timeout=timeout)
        self.api_key = os.environ["MODEL_API_KEY"]

    @staticmethod
    def _encode_png(rgb: np.ndarray) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def segment(self, rgb, prompts):
        resp = self.client.post(
            f"{self.base_url}/segment",   # TODO: verify endpoint
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "image": self._encode_png(rgb), "prompts": prompts},
        )
        resp.raise_for_status()
        out = {}
        for item in resp.json().get("results", []):   # TODO: verify response shape
            m = _decode_mask(item["mask"], rgb.shape[:2])
            out[item["prompt"]] = Mask(item["prompt"], m, float(item.get("score", 0.0)))
        return out


def _decode_mask(data, shape) -> np.ndarray:
    # TODO: implement for the actual encoding returned by the API.
    raise NotImplementedError("SAM 3.1 mask decoding not wired up yet")
