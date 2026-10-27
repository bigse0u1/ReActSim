"""Side-by-side video for Show, don't tell: the human demonstration (with SAM 3.1 masks)
on the left, the robot doing the task on the right, and what was learned in between."""

from __future__ import annotations

from pathlib import Path

import numpy as np

HAND, TARGET, REFERENCE, OTHER = (230, 70, 70), (60, 170, 255), (90, 220, 110), (200, 200, 200)
W, H = 1280, 720
FPS = 30


def overlay_frames(frames: list[np.ndarray], hand, objects, target, reference) -> list[np.ndarray]:
    """Colour each tracked mask: hand red, moved object blue, destination green."""
    out = []
    for i, f in enumerate(frames):
        im = f.astype(np.float32)
        for t in objects + hand:
            m = t.masks.get(i)
            if m is None:
                continue
            col = HAND if t in hand else TARGET if t is target else REFERENCE if t is reference else OTHER
            im[m] = im[m] * 0.45 + np.array(col) * 0.55
        out.append(im.astype(np.uint8))
    return out


def write_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    import imageio.v2 as imageio

    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)
    return Path(path)


def _read(path: str | Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    return [f for f in imageio.get_reader(str(path))]


def _fit(im: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    s = min(w / im.shape[1], h / im.shape[0])
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def _text(im: np.ndarray, lines: list[tuple[str, tuple]], x: int, y: int, scale=0.6, gap=28) -> None:
    import cv2

    for i, (s, col) in enumerate(lines):
        cv2.putText(im, s, (x, y + i * gap), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def reason(c: dict) -> str:
    """A failure in words a viewer gets at a glance."""
    d = c.get("detail", "")
    if c["failure"] == "goal_not_met":
        return ("tipped over" if "upright" in d else "missed the plate" if "over_reference" in d
                else "not placed")
    return {"controller_error": d or "no feasible motion", "grasp_failed": "slipped",
            "object_moved": "knocked it away"}.get(c["failure"], c["failure"])


def compose(human_video: str | Path, robot_video: str | Path, demo, imagined: list[dict],
            used_grasp: str, success: bool, out: str | Path, human_fps: float,
            body: str = "two-finger gripper") -> Path:
    """Phase 1: the human clip plays while SAM tracks it. Phase 2: the robot plays, with
    what was learned from the human and what Imagination found next to it."""
    human = _read(human_video)
    robot = _read(robot_video)
    repeat = max(int(round(FPS / human_fps)), 1)
    white, grey, blue, green, red = (255, 255, 255), (170, 170, 170), TARGET, REFERENCE, HAND

    learned = [("WATCHED (SAM 3.1 + Muse Spark)", white),
               (f"moved: {demo.target}  ->  onto: {demo.reference}", blue),
               (f"human grasp: {demo.human_grasp}", red)]
    first = next((i for i in imagined if i["proposed"].startswith("pick(")), None)
    imagine = [("IMAGINED (physics, pick + place)", white)]
    for c in (first or {}).get("candidates", []):
        grasp = c["action"].split("[")[-1].rstrip("]")
        ok = c["failure"] == "none"
        imagine.append((f"{grasp:<7} {'works' if ok else reason(c)}", green if ok else red))
    verdict = (f"robot grasp: {used_grasp}" + ("" if used_grasp == demo.human_grasp
                                               else f"  (adapted from human {demo.human_grasp})"))
    result = [(verdict, white), ("SUCCESS: on the plate" if success else "FAILED", green if success else red)]

    frames = []
    left_w, right_w = 640, 640
    for f in [h for h in human for _ in range(repeat)]:
        c = np.zeros((H, W, 3), np.uint8)
        c[:360, :left_w] = _fit(f, left_w, 360)
        c[:, left_w:] = _fit(robot[0], right_w, H)
        _text(c, [("1. HUMAN DEMONSTRATION (phone video)", white),
                  ("red: hand   blue: object moved   green: destination", grey)], 20, 400)
        frames.append(c)
    last = _fit(human[-1], left_w, 360)
    for i, r in enumerate(robot):
        c = np.zeros((H, W, 3), np.uint8)
        c[:360, :left_w] = last
        c[:, left_w:] = _fit(r, right_w, H)
        lines = learned + [("", white)] + imagine + [("", white)]
        if i > len(robot) // 3:
            lines += result
        _text(c, [(f"2. ROBOT ({body})", white)], left_w + 20, 40)
        _text(c, lines, 20, 395, scale=0.55, gap=24)
        frames.append(c)
    frames += [frames[-1]] * FPS * 2
    return write_video(frames, out, FPS)
