"""Shared data types passed between perception, planning, control and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class ObjectState:
    name: str                      # stable id, e.g. "banana"
    position: np.ndarray           # world xyz (m)
    color: str = ""
    kind: str = ""                 # banana | mug | can | plate | ...
    held: bool = False
    source: str = "gt"             # gt | depth (how position was estimated)
    confidence: float = 1.0
    prompt: str = ""               # SAM noun phrase; defaults to the name

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "color": self.color,
            "position": [round(float(v), 3) for v in self.position],
            "held": self.held,
        }


@dataclass
class Observation:
    """Raw sensor data. rgb/depth are None for worlds without rendering (FakeWorld)."""
    rgb: np.ndarray | None = None          # HxWx3 uint8
    depth: np.ndarray | None = None        # HxW float meters
    intrinsic: np.ndarray | None = None    # 3x3 OpenCV K
    extrinsic: np.ndarray | None = None    # 3x4 world->camera (OpenCV)
    segmentation: np.ndarray | None = None # HxW int actor ids (oracle only)
    gt_objects: dict[str, ObjectState] = field(default_factory=dict)
    gripper_holding: str | None = None


@dataclass
class WorldState:
    objects: dict[str, ObjectState]
    holding: str | None
    step: int

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "holding": self.holding,
            "objects": [o.to_prompt_dict() for o in self.objects.values()],
        }


GRASP_MODES = ("top", "angled", "side")   # straight down / 50° from the robot side / horizontal


class ActionType(str, Enum):
    PICK = "pick"
    PLACE_ON = "place_on"
    PUSH_ASIDE = "push_aside"
    REOBSERVE = "reobserve"
    HOME = "home"
    DONE = "done"
    GIVE_UP = "give_up"


@dataclass
class Action:
    type: ActionType
    target: str | None = None       # object to act on
    reference: str | None = None    # place_on destination
    grasp: str = "top"              # one of GRASP_MODES
    rationale: str = ""
    # Where a non-top grasp comes from: degrees around the object, 0 = from the robot's side.
    # Chosen by Imagination, not by the planner (None means 0).
    approach_deg: float | None = None

    def short(self) -> str:
        args = ", ".join(a for a in (self.target, self.reference) if a)
        grasp = self.grasp
        if self.approach_deg and self.grasp != "top":
            grasp += f"@{self.approach_deg:+.0f}"
        extra = f" [{grasp}]" if self.type == ActionType.PICK else ""
        return f"{self.type.value}({args}){extra}"


@dataclass
class Goal:
    target: str
    relation: str      # "on"
    reference: str


@dataclass
class Plan:
    goal: Goal
    actions: list[Action]
    rationale: str = ""


@dataclass
class SkillResult:
    ok: bool                 # controller reports it finished without planner/IK error
    message: str = ""
    duration_s: float = 0.0


class FailureType(str, Enum):
    NONE = "none"
    GRASP_FAILED = "grasp_failed"
    DROPPED = "dropped"
    OBJECT_MOVED = "object_moved"
    PLACE_MISSED = "place_missed"
    TARGET_LOST = "target_lost"
    CONTROLLER_ERROR = "controller_error"
    GOAL_NOT_MET = "goal_not_met"      # every step "worked" but the success check still fails
    NO_WORKING_GRASP = "no_working_grasp"   # Imagination: every candidate fails and none even moves the object


@dataclass
class FailureReport:
    type: FailureType
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.type != FailureType.NONE


@dataclass
class PerturbEvent:
    kind: str                         # teleport | drop | knock
    object: str | None = None
    dx: float = 0.0
    dy: float = 0.0
    source: str = "scripted"          # scripted | human | adversary
