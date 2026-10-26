"""Live dashboard via rerun.io: camera, SAM masks, plan, failures, imagined futures.

No-op when rerun-sdk is not installed or disabled, so the core loop never depends on it.
"""

from __future__ import annotations

import numpy as np


class Dashboard:
    def __init__(self, enabled: bool = False, app_id: str = "ReActSim"):
        self.rr = None
        if enabled:
            try:
                import rerun as rr

                rr.init(app_id, spawn=True)
                self.rr = rr
            except ImportError:
                print("[viz] rerun-sdk not installed; dashboard disabled")

    def _t(self, step):
        if self.rr:
            self.rr.set_time_sequence("step", step)

    def observation(self, step, obs, masks=None):
        if not self.rr or obs.rgb is None:
            return
        self._t(step)
        self.rr.log("camera/rgb", self.rr.Image(obs.rgb))
        if masks:
            seg = np.zeros(obs.rgb.shape[:2], dtype=np.uint16)
            for i, m in enumerate(masks.values(), start=1):
                seg[m.mask] = i
            self.rr.log("camera/sam", self.rr.SegmentationImage(seg))

    def objects(self, step, state):
        if not self.rr:
            return
        self._t(step)
        names = list(state.objects)
        pts = np.array([state.objects[n].position for n in names])
        self.rr.log("world/objects", self.rr.Points3D(pts, labels=names, radii=0.015))

    def text(self, step, channel, msg):
        print(f"[{step:02d}] {channel:<10} {msg}")
        if self.rr:
            self._t(step)
            self.rr.log(f"log/{channel}", self.rr.TextLog(msg))

    def imagined(self, step, imagined):
        for k, im in enumerate(imagined):
            self.text(step, "imagine", f"#{k} {im.action.short():<28} score={im.score:+.2f} "
                                       f"{im.failure.type.value}")
            if self.rr and im.frames:
                self.rr.log(f"imagine/{k}", self.rr.Image(im.frames[-1]))
