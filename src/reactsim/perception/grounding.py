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


def mask_points(mask: np.ndarray, depth: np.ndarray, K: np.ndarray, extrinsic: np.ndarray,
                erode: int = 2) -> np.ndarray:
    """World-frame points of the masked pixels. The mask is eroded first: pixels on the
    silhouette mix object and background depth ("flying pixels")."""
    import cv2

    m = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)).astype(bool)
    vs, us = np.nonzero(m & (depth > 0))
    z = depth[vs, us]
    pts_cam = np.stack([(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return (pts_cam - t) @ R


def estimate_box(points: np.ndarray, table_z: float = 0.0, min_points: int = 30):
    """Oriented box of an object resting on the table, from the points the camera sees.

    The camera only sees the top and front of an object, so the plain centroid is pulled
    toward the camera. The top surface is seen whole from above, so its centroid gives the
    centre in x-y; the object stands on the table, so its height is the highest point.
    The long axis comes from the spread of all points in x-y (matters for a banana).
    Returns a trimesh Box (same type as ManiSkill's actor OBB) or None.
    """
    import trimesh

    pts = points[points[:, 2] > table_z + 0.004]          # drop the table
    if len(pts) < min_points:
        return None
    zmax = np.percentile(pts[:, 2], 98)                   # robust to stray pixels
    top = pts[pts[:, 2] > zmax - 0.012]
    c_xy = top[:, :2].mean(axis=0)
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    major = np.linalg.eigh(xy.T @ xy)[1][:, -1]
    minor = np.array([-major[1], major[0]])
    d = pts[:, :2] - c_xy
    length = 2 * np.percentile(np.abs(d @ major), 98)
    width = 2 * np.percentile(np.abs(d @ minor), 98)     # the far half is hidden: mirror the near one
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = [*major, 0], [*minor, 0], [0, 0, 1]
    T[:3, 3] = [*c_xy, table_z + zmax / 2]
    return trimesh.primitives.Box(extents=[length, width, zmax - table_z], transform=T)
