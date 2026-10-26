"""A tiny geometric stand-in for ManiSkill.

It lets the whole agent loop (planning, diagnosis, imagination, chaos, metrics)
be developed and unit-tested without a GPU. It is NOT a physics simulator and
results from it must never be reported as robot results.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from reactsim.types import (
    Action,
    ActionType,
    Goal,
    Observation,
    ObjectState,
    PerturbEvent,
    SkillResult,
)

TABLE_Z = 0.02
PLATE_HALF = 0.06
REACH_RADIUS = 0.45   # from robot base at origin


@dataclass
class _FakeState:
    objects: dict[str, ObjectState]
    holding: str | None
    # object -> grasp modes that will slip, to exercise failure/imagination paths
    slippery: dict[str, set[str]]


class FakeWorld:
    def __init__(self, slippery: dict[str, set[str]] | None = None):
        self._slippery_init = slippery or {}
        self.state: _FakeState | None = None

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)

        def jitter(xy):
            return np.array([*(np.array(xy) + rng.uniform(-0.03, 0.03, 2)), TABLE_Z])

        objects = {
            "red_cube": ObjectState("red_cube", jitter([0.10, 0.15]), "red", "cube"),
            "blue_cube": ObjectState("blue_cube", jitter([0.15, -0.05]), "blue", "cube"),
            "green_cube": ObjectState("green_cube", jitter([0.00, -0.20]), "green", "cube"),
            "plate": ObjectState("plate", np.array([0.25, 0.20, 0.005]), "white", "plate"),
        }
        self.state = _FakeState(objects, None, copy.deepcopy(self._slippery_init))

    def observe(self) -> Observation:
        s = self.state
        objs = copy.deepcopy(s.objects)
        for o in objs.values():
            o.held = o.name == s.holding
        return Observation(gt_objects=objs, gripper_holding=s.holding)

    def execute(self, action: Action) -> SkillResult:
        s = self.state
        t = action.type
        if t == ActionType.PICK:
            obj = s.objects.get(action.target)
            if obj is None:
                return SkillResult(False, f"unknown object {action.target}")
            if np.linalg.norm(obj.position[:2]) > REACH_RADIUS:
                return SkillResult(False, "target out of reach (IK failed)")
            if s.holding is not None:
                return SkillResult(False, "gripper already holding")
            # Controller "succeeds" even when the grasp slips: diagnosis must catch it.
            if action.grasp not in s.slippery.get(obj.name, set()):
                s.holding = obj.name
                obj.position = obj.position + np.array([0, 0, 0.15])
            return SkillResult(True, "pick motion finished", 4.0)
        if t == ActionType.PLACE_ON:
            if s.holding is None:
                return SkillResult(False, "nothing in gripper")
            ref = s.objects[action.reference]
            obj = s.objects[s.holding]
            obj.position = np.array([ref.position[0], ref.position[1], ref.position[2] + 0.025])
            s.holding = None
            return SkillResult(True, "place motion finished", 4.0)
        if t == ActionType.PUSH_ASIDE:
            obj = s.objects[action.target]
            obj.position = obj.position + np.array([0.0, 0.1, 0.0])
            return SkillResult(True, "push finished", 3.0)
        if t in (ActionType.REOBSERVE, ActionType.HOME, ActionType.DONE, ActionType.GIVE_UP):
            return SkillResult(True, t.value, 0.5)
        return SkillResult(False, f"unsupported action {t}")

    def perturb(self, event: PerturbEvent) -> None:
        s = self.state
        if event.kind == "drop" and s.holding:
            obj = s.objects[s.holding]
            obj.position = np.array([obj.position[0] + 0.03, obj.position[1], TABLE_Z])
            s.holding = None
        elif event.kind in ("teleport", "knock") and event.object in s.objects:
            if s.holding == event.object:
                s.holding = None
            obj = s.objects[event.object]
            obj.position = np.array([obj.position[0] + event.dx, obj.position[1] + event.dy, TABLE_Z])

    def is_goal_satisfied(self, goal: Goal) -> bool:
        s = self.state
        obj, ref = s.objects.get(goal.target), s.objects.get(goal.reference)
        if obj is None or ref is None or s.holding == goal.target:
            return False
        dxy = np.abs(obj.position[:2] - ref.position[:2])
        return bool(np.all(dxy < PLATE_HALF) and obj.position[2] > ref.position[2])

    def snapshot(self):
        return copy.deepcopy(self.state)

    def restore(self, snap) -> None:
        self.state = copy.deepcopy(snap)

    def frame(self):
        return None
