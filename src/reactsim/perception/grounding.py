"""Mask + depth -> 3D object position, and the perception step that builds WorldState."""

from __future__ import annotations

import copy

import numpy as np

from reactsim.perception.segmenter import OracleSegmenter, Segmenter
from reactsim.types import Observation, ObjectState, WorldState


def backproject_mask(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                     extrinsic: np.ndarray, min_pixels: int = 20) -> np.ndarray | None:
    """Median world-frame point of the masked pixels (OpenCV camera convention).

    extrinsic is 3x4 world->camera: X_cam = R @ X_world + t.
    """
    vs, us = np.nonzero(mask & (depth > 0))
    if len(us) < min_pixels:
        return None
    z = depth[vs, us]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=1)
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    pts_world = (pts_cam - t) @ R          # R^T (X_cam - t), row-vector form
    return np.median(pts_world, axis=0)


def prompt_for(obj: ObjectState) -> str:
    return obj.prompt or obj.name.replace("_", " ")


class Perception:
    """Builds the WorldState the planner sees.

    Identity of objects comes from the segmenter (that is the part SAM 3.1 does).
    Position comes from depth back-projection when a mask exists, else GT. Every
    estimate records its source and the error vs GT so the gap can be reported.
    """

    def __init__(self, segmenter: Segmenter | None = None, use_depth: bool = True,
                 look_every_step: bool = True):
        self.segmenter = segmenter
        self.use_depth = use_depth
        # SAM 3.1 costs ~2 s per object, so the agent only "looks" at the start, after a
        # failure and on reobserve; in between (and inside Imagination) state comes from the sim.
        self.look_every_step = look_every_step
        self.last_errors: dict[str, float] = {}
        self.last_masks = {}

    def perceive(self, obs: Observation, step: int, look: bool = False) -> WorldState:
        objects = copy.deepcopy(obs.gt_objects)
        self.last_errors, self.last_masks = {}, {}
        if self.segmenter is None or obs.rgb is None or not (look or self.look_every_step):
            return WorldState(objects, obs.gripper_holding, step)

        if isinstance(self.segmenter, OracleSegmenter):
            self.segmenter.segmentation = obs.segmentation
        masks = self.segmenter.segment(obs.rgb, [prompt_for(o) for o in objects.values()])
        self.last_masks = masks
        for obj in objects.values():
            m = masks.get(prompt_for(obj))
            if m is None:
                obj.confidence = 0.0
                obj.source = "gt(unseen)"
                continue
            obj.confidence = m.score
            if self.use_depth and obs.depth is not None and not obj.held:
                p = backproject_mask(m.mask, obs.depth, obs.intrinsic, obs.extrinsic)
                if p is not None:
                    self.last_errors[obj.name] = float(np.linalg.norm(p - obj.position))
                    obj.position = p
                    obj.source = "depth"
        return WorldState(objects, obs.gripper_holding, step)
