"""What happened in a human demonstration, measured from SAM 3.1 video tracks.

Geometry answers the questions it can answer reliably: which object moved, when
the hand took it, when it let go, and what it ended up on. How the hand grasped
it is left to Muse Spark looking at the grasp frames (see demonstration.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reactsim.perception.video import Track

MOVED_FRAC = 0.10    # an object moved if its centroid travelled this fraction of the frame width
LIFT_FRAC = 0.03     # ... and it started moving once it left its start by this much
STILL_FRAC = 0.01    # it is at rest once it stays within this much of its final position
CONTACT_DILATE = 5   # px; hand and object masks this close count as touching


@dataclass
class DemoEvents:
    target: Track
    reference: Track | None
    grasp_frame: int        # last frame before the object starts moving (hand on it)
    release_frame: int      # first frame the object is at rest at its destination
    displacement_px: float
    contact_frames: list[int]

    def summary(self) -> dict:
        return {"target": self.target.prompt,
                "reference": self.reference.prompt if self.reference else None,
                "grasp_frame": self.grasp_frame, "release_frame": self.release_frame,
                "displacement_px": round(self.displacement_px, 1),
                "contact_frames": [self.contact_frames[0], self.contact_frames[-1]]
                if self.contact_frames else []}


def _path(t: Track, n: int) -> list[np.ndarray | None]:
    return [t.centroid(i) for i in range(n)]


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def bottom_point(mask: np.ndarray) -> np.ndarray:
    """Where an object rests: the lowest mask row, at the mask's mean column there."""
    ys, xs = np.nonzero(mask)
    y = ys.max()
    return np.array([xs[ys >= y - 2].mean(), y])


def extract_events(hand: list[Track], objects: list[Track], n_frames: int, width: int) -> DemoEvents:
    """Find the one object the hand moved and the moments it was picked and put down."""
    def first_last(path):
        seen = [p for p in path if p is not None]
        return (seen[0], seen[-1]) if seen else (None, None)

    moved = []
    for t in objects:
        a, b = first_last(_path(t, n_frames))
        if a is not None:
            moved.append((float(np.linalg.norm(b - a)), t))
    if not moved:
        raise ValueError("no object was tracked")
    disp, target = max(moved, key=lambda m: m[0])
    if disp < MOVED_FRAC * width:
        raise ValueError(f"nothing moved: largest displacement {disp:.0f}px ({target.prompt})")

    path = _path(target, n_frames)
    start, end = first_last(path)
    lift = next(i for i, p in enumerate(path) if p is not None and np.linalg.norm(p - start) > LIFT_FRAC * width)
    last_moving = max(i for i, p in enumerate(path) if p is not None and np.linalg.norm(p - end) > STILL_FRAC * width)
    release = min(last_moving + 1, n_frames - 1)

    contact = []
    for i in range(n_frames):
        om = target.masks.get(i)
        if om is None:
            continue
        if any(h.masks.get(i) is not None and (_dilate(h.masks[i], CONTACT_DILATE) & om).any() for h in hand):
            contact.append(i)
    before = [i for i in contact if i < lift]
    grasp = before[-1] if before else max(lift - 1, 0)

    # The destination is the other object whose mask holds the target's resting point.
    last = max(target.masks)
    rest = bottom_point(target.masks[last]).astype(int)
    reference = None
    for _, t in sorted(moved, key=lambda m: m[0]):
        if t is target or last not in t.masks:
            continue
        m = _dilate(t.masks[last], CONTACT_DILATE)
        if m[min(rest[1], m.shape[0] - 1), min(rest[0], m.shape[1] - 1)]:
            reference = t
            break
    return DemoEvents(target, reference, grasp, release, disp, contact)
