"""SAM 3.1 video tracking for human demonstration clips.

The clip is downsampled (fps, width) before upload: SAM bills per frame, and a
pick-and-place needs ~8 fps, not 30. One upload is reused for every prompt.
See https://dev.meta.ai/docs/sam/segmenting (video, Files API).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reactsim.perception.segmenter import parse_sam_video


@dataclass
class Track:
    """One object (one SAM object id of one prompt) through the clip."""
    prompt: str
    object_id: str = "0"
    masks: dict[int, np.ndarray] = field(default_factory=dict)   # frame -> HxW bool
    scores: dict[int, float] = field(default_factory=dict)

    def centroid(self, frame: int) -> np.ndarray | None:
        m = self.masks.get(frame)
        if m is None or not m.any():
            return None
        ys, xs = np.nonzero(m)
        return np.array([xs.mean(), ys.mean()])


def downsample(src: str | Path, dst: str | Path, fps: float = 8, width: int = 640) -> Path:
    """Re-encode to H.264 at a lower frame rate and size (SAM accepts H.264 MP4)."""
    import imageio_ffmpeg

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(src),
                    "-vf", f"fps={fps},scale={width}:-2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    return dst


def read_frames(path: str | Path) -> list[np.ndarray]:
    """All frames as RGB uint8, in the same order SAM indexes them."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return frames


class SamVideoTracker:
    def __init__(self, model: str = "sam-3.1", base_url: str | None = None, timeout: float = 300.0):
        import httpx

        self.model = model
        self.base_url = base_url or os.environ.get("META_API_BASE_URL", "https://api.meta.ai/v1")
        self.api_key = os.environ["MODEL_API_KEY"]   # never log this
        self.client = httpx.Client(timeout=timeout)
        self.frames_billed = 0
        self.calls: list[float] = []

    @property
    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def upload(self, path: str | Path) -> str:
        with open(path, "rb") as f:
            r = self.client.post(f"{self.base_url}/files", headers=self._auth,
                                 data={"purpose": "user_data"},
                                 files={"file": (Path(path).name, f, "video/mp4")})
        if r.status_code >= 400:
            raise RuntimeError(f"upload failed ({r.status_code}): {r.text[:300]}")
        return r.json()["id"]

    def _request(self, file_id: str, prompt: str) -> str:
        t0, text = time.time(), []
        body = {"model": self.model, "stream": True,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_video", "file_id": file_id}]}],
                "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"}}
        with self.client.stream("POST", f"{self.base_url}/responses", headers=self._auth, json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"SAM video request failed ({r.status_code}): {r.text[:300]}")
            for line in r.iter_lines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                e = json.loads(line[5:])
                if e.get("type") == "response.output_text.delta":
                    text.append(e["delta"])
                elif e.get("type") == "response.completed":
                    usage = e["response"].get("usage") or {}
                    self.frames_billed += int(usage.get("video_frames_processed", 0))
                elif e.get("type") in ("error", "response.failed"):
                    raise RuntimeError(f"SAM video request failed: {json.dumps(e)[:300]}")
        self.calls.append(time.time() - t0)
        return "".join(text)

    def track(self, video: str | Path, prompts: list[str],
              frame_hw: tuple[int, int]) -> dict[str, list[Track]]:
        """prompt -> one Track per object SAM found for it (e.g. two cans -> two tracks)."""
        file_id = self.upload(video)
        tracks = {}
        for p in prompts:
            by_id: dict[str, Track] = {}
            for frame, objs in parse_sam_video(self._request(file_id, p), frame_hw).items():
                for obj, mask, conf in objs:
                    t = by_id.setdefault(obj, Track(p, obj))
                    t.masks[frame], t.scores[frame] = mask, conf
            tracks[p] = list(by_id.values())
        return tracks
