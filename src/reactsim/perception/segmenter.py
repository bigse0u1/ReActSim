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

    def __init__(self, seg_ids: dict[str, int], aliases: dict[str, str] | None = None):
        self.seg_ids = seg_ids
        self.aliases = aliases or {}   # SAM prompt -> actor name, e.g. "cup" -> "mug"
        self.segmentation: np.ndarray | None = None   # set per frame by the perception module

    def segment(self, rgb, prompts):
        out = {}
        for p in prompts:
            name = self.aliases.get(p, p.strip().lower().replace(" ", "_"))
            if name in self.seg_ids and self.segmentation is not None:
                m = self.segmentation == self.seg_ids[name]
                if m.any():
                    out[p] = Mask(p, m, 1.0)
        return out


class Sam3Segmenter:
    """SAM 3.1 via the Meta Model API (POST /v1/responses, one noun phrase per request).

    The response is SAM's special-token text (box + mask records), decoded with
    Meta's `meta-sam-parser`. See https://dev.meta.ai/docs/sam/segmenting.
    """

    def __init__(self, model: str = "sam-3.1", base_url: str | None = None, timeout: float = 60.0):
        import httpx

        self.model = model
        self.base_url = base_url or os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1")
        self.client = httpx.Client(timeout=timeout)
        self.api_key = os.environ["MODEL_API_KEY"]   # never log this
        self.calls: list[float] = []                 # latency per request

    @staticmethod
    def _data_url(rgb: np.ndarray) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _request(self, rgb: np.ndarray, prompt: str) -> str:
        import time

        t0 = time.time()
        resp = self.client.post(
            f"{self.base_url}/responses",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": self._data_url(rgb)},
                ]}],
                "metadata": {"include_confidence": "true"},
            },
        )
        self.calls.append(time.time() - t0)
        if resp.status_code >= 400:
            raise RuntimeError(f"SAM request failed ({resp.status_code}): {resp.text[:300]}")
        return _output_text(resp.json())

    def segment(self, rgb, prompts):
        out = {}
        for p in prompts:
            masks = parse_sam_masks(self._request(rgb, p), rgb.shape[:2])
            if masks:
                mask, score = max(masks, key=lambda m: (m[1], m[0].sum()))
                out[p] = Mask(p, mask, score)
        return out


def _output_text(data: dict) -> str:
    if data.get("output_text"):
        return data["output_text"]
    parts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "\n".join(parts)


def parse_sam_masks(text: str, frame_hw: tuple[int, int]) -> list[tuple[np.ndarray, float]]:
    """SAM output text -> [(full-frame bool mask, confidence)] for one image."""
    from meta_sam_parser import image_segmentation_format

    return [(m, c) for _, _, m, c in _decode(image_segmentation_format(), text, frame_hw)]


def parse_sam_video(text: str, frame_hw: tuple[int, int]) -> dict[int, list[tuple[str, np.ndarray, float]]]:
    """SAM video output text -> {frame index: [(object id, full-frame bool mask, confidence)]}."""
    from meta_sam_parser import video_segmentation_format

    out: dict[int, list] = {}
    for frame, obj, mask, conf in _decode(video_segmentation_format(), text, frame_hw):
        out.setdefault(frame, []).append((obj, mask, conf))
    return out


def _decode(fmt, text: str, frame_hw: tuple[int, int]):
    """Yield (frame index or None, object id, full-frame bool mask, confidence) per mask record.
    SAM masks are box-local and may be coarser than their box."""
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster

    parser = fmt.create_parser()
    parser.push(text + "\n", emit=False)
    result = parser.finish(CompletedOutcome()).result
    H, W = frame_hw
    for rec in result.records:
        if rec.kind != "mask":
            continue
        m = rec.mask
        local = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
        b = rec.bounds
        top, left = int(round(b.top)), int(round(b.left))
        bh, bw = int(round(b.bottom)) - top, int(round(b.right)) - left
        if (bh, bw) != local.shape and bh > 0 and bw > 0:
            rows = (np.arange(bh) * local.shape[0] / bh).astype(int)
            cols = (np.arange(bw) * local.shape[1] / bw).astype(int)
            local = local[rows][:, cols]
        full = np.zeros((H, W), dtype=bool)
        r0, c0 = max(top, 0), max(left, 0)
        r1, c1 = min(top + local.shape[0], H), min(left + local.shape[1], W)
        full[r0:r1, c0:c1] = local[r0 - top:r1 - top, c0 - left:c1 - left].astype(bool)
        frame = rec.frame.frame_index if rec.frame is not None else None
        yield frame, rec.object_id, full, rec.confidence if rec.confidence is not None else 1.0
