"""Show, don't tell: turn a phone video of a human doing a task into a robot instruction.

    Muse Spark   first + last frame  -> the objects on the table, as SAM noun phrases
    SAM 3.1      video tracking      -> hand and object masks in every frame
    geometry     tracks              -> what moved, when it was grasped, where it went
    Muse Spark   grasp-moment crops  -> how the hand grasped it (top / angled / side)

The result is intent, not a trajectory: the robot has two fingers, the human five.
Which grasp the robot uses is decided later by Imagination, with the human's as a hint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from reactsim.imitation.events import DemoEvents, extract_events
from reactsim.perception.video import SamVideoTracker, Track, downsample, read_frames
from reactsim.planning.schema import extract_json
from reactsim.types import GRASP_MODES

BODIES = {"panda": "a two-finger parallel gripper",
          "hand": "a five-finger robot hand (xArm7 + Ability hand)"}
SAMPLE_FPS = 8      # SAM bills per frame; 8 fps is plenty for a pick-and-place
SAMPLE_WIDTH = 640

NAME_PROMPT = """\
You see the first and the last frame of a short video in which a person does a task
on a table. List the objects on the table (not the person's hand or arm).
For each object give:
- "phrase": a short noun phrase a segmentation model can find (1-2 words, e.g. "can", "paper cup")
- "fallback": a more generic single word in case the phrase is not found (e.g. "fruit", "cup")
- "robot_object": which of the robot's objects it corresponds to, from this list: {robot_objects};
  null if none fits. Match by function and shape, not colour.
Answer only with JSON: {{"objects": [{{"phrase": ..., "fallback": ..., "robot_object": ...}}]}}"""

GRASP_PROMPT = """\
These crops show a person's hand grasping the {target} and starting to lift it
(in time order). Classify how the hand grasps it, using a robot's grasp vocabulary:
- "top": the hand comes from above; fingers point down onto the object
- "side": the hand comes from the side; palm roughly vertical, fingers wrap around the object
- "angled": in between, the hand comes diagonally from above
Answer only with JSON:
{{"grasp": "top" | "angled" | "side",
  "evidence": "<one sentence on what the hand does>"}}"""


@dataclass
class Demonstration:
    video: str
    target: str                  # SAM phrase of the moved object
    reference: str | None        # SAM phrase of what it was put on
    robot_target: str | None     # matching objects in the robot's scene
    robot_reference: str | None
    human_grasp: str             # one of GRASP_MODES
    grasp_evidence: str
    grasp_frame: int
    release_frame: int
    objects: list[dict] = field(default_factory=list)
    sam_frames_billed: int = 0
    sam_overlay: str = ""        # the sampled clip with SAM masks drawn on it

    @property
    def task(self) -> str:
        return f"The person picked up the {self.target} and put it on the {self.reference}."

    def instruction(self, body: str = "a two-finger parallel gripper") -> str:
        return (f"Put the {self.robot_target} on the {self.robot_reference}. "
                f"This task was shown in a human demonstration. {self.task} "
                f"The human used {'an' if self.human_grasp[0] in 'aeiou' else 'a'} {self.human_grasp} grasp: {self.grasp_evidence} "
                f"The robot has {body}, not a human hand; "
                "use the human's grasp as a hint, not a rule.")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Demonstration":
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _crop(frame: np.ndarray, masks: list[np.ndarray], pad: float = 0.35) -> np.ndarray:
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    h, w = ys.max() - ys.min(), xs.max() - xs.min()
    y0, y1 = max(int(ys.min() - pad * h), 0), min(int(ys.max() + pad * h), frame.shape[0])
    x0, x1 = max(int(xs.min() - pad * w), 0), min(int(xs.max() + pad * w), frame.shape[1])
    return frame[y0:y1, x0:x1]


def _hi_res_frame(video: Path, index: int, fps: float) -> np.ndarray:
    """The original-resolution frame at sampled index `index` (crops need the detail)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(index * cap.get(cv2.CAP_PROP_FPS) / fps))
    ok, im = cap.read()
    if not ok:
        raise ValueError(f"cannot read frame {index} of {video}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def name_objects(muse, first: np.ndarray, last: np.ndarray, robot_objects: list[str]) -> list[dict]:
    text = muse.chat("You describe scenes for a robot. Be literal and brief.",
                     NAME_PROMPT.format(robot_objects=robot_objects), images=[first, last])
    objs = extract_json(text).get("objects", [])
    return [o for o in objs if o.get("phrase")]


def track_objects(tracker: SamVideoTracker, video: Path, objects: list[dict],
                  hw: tuple[int, int]) -> tuple[list[Track], list[Track]]:
    """Track the hand and each object; retry an object with its fallback word if SAM finds nothing."""
    prompts = ["hand"] + [o["phrase"] for o in objects]
    tracks = tracker.track(video, prompts, hw)
    for o in objects:
        if not tracks.get(o["phrase"]) and o.get("fallback") and o["fallback"] != o["phrase"]:
            tracks.update(tracker.track(video, [o["fallback"]], hw))
            if tracks[o["fallback"]]:
                o["phrase"] = o["fallback"]
    objs = [t for o in objects for t in tracks.get(o["phrase"], [])]
    return tracks.get("hand", []), objs


def describe_grasp(muse, video: Path, ev: DemoEvents, hand: list[Track], fps: float) -> dict:
    import cv2

    crops = []
    for i in (ev.grasp_frame - 2, ev.grasp_frame, ev.grasp_frame + 3):
        i = int(np.clip(i, 0, max(ev.target.masks)))
        masks = [m for m in [ev.target.masks.get(i)] + [h.masks.get(i) for h in hand] if m is not None]
        frame = _hi_res_frame(video, i, fps)
        big = [cv2.resize(m.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
        crops.append(_crop(frame, big))
    d = extract_json(muse.chat("You analyse human grasps for a robot. Be literal and brief.",
                               GRASP_PROMPT.format(target=ev.target.prompt), images=crops))
    if d.get("grasp") not in GRASP_MODES:
        raise ValueError(f"Muse gave no valid grasp: {d}")
    return d


def learn_from_video(video: str | Path, muse, tracker: SamVideoTracker, robot_objects: list[str],
                     work_dir: str | Path) -> Demonstration:
    video = Path(video)
    small = downsample(video, Path(work_dir) / f"{video.stem}_{SAMPLE_FPS}fps.mp4", SAMPLE_FPS, SAMPLE_WIDTH)
    frames = read_frames(small)
    objects = name_objects(muse, frames[0], frames[-1], robot_objects)
    hand, objs = track_objects(tracker, small, objects, frames[0].shape[:2])
    ev = extract_events(hand, objs, len(frames), frames[0].shape[1])
    grasp = describe_grasp(muse, video, ev, hand, SAMPLE_FPS)
    robot = {o["phrase"]: o.get("robot_object") for o in objects}
    demo = Demonstration(
        video=str(video), target=ev.target.prompt,
        reference=ev.reference.prompt if ev.reference else None,
        robot_target=robot.get(ev.target.prompt),
        robot_reference=robot.get(ev.reference.prompt) if ev.reference else None,
        human_grasp=grasp["grasp"], grasp_evidence=grasp.get("evidence", ""),
        grasp_frame=ev.grasp_frame, release_frame=ev.release_frame,
        objects=objects, sam_frames_billed=tracker.frames_billed)
    from reactsim.viz.imitation_video import overlay_frames, write_video
    overlay = overlay_frames(frames, hand, objs, ev.target, ev.reference)
    demo.sam_overlay = str(write_video(overlay, Path(work_dir) / "human_sam.mp4", SAMPLE_FPS))
    return demo
