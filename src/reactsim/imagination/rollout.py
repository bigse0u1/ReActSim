"""Imagination: try candidate actions in a forked copy of the physics, then commit to the best.

This is real lookahead (the simulator steps the candidate forward and is restored),
not a heuristic feasibility score. On a real robot the same idea needs a
digital twin; in the demo we say exactly that.

Note: if the world is wrapped in RecordEpisode, imagined rollouts are recorded
too. Either pause recording or use those frames as the "ghost" split-screen.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from reactsim.evaluation.diagnosis import diagnose, goal_not_met
from reactsim.perception.grounding import Perception
from reactsim.types import Action, FailureReport, FailureType, Goal, SkillResult, WorldState


MAX_NUDGES_IMAGINED = 2   # nudging candidates imagined one step further when nothing works


@dataclass
class Imagined:
    action: Action
    result: SkillResult
    failure: FailureReport
    score: float
    frames: list = field(default_factory=list)
    trace: Any = None        # world-specific record that lets the real run replay this exact rollout
    then: Action | None = None   # follow-up imagined after the action (e.g. the place after a pick)
    then_trace: Any = None
    # For a candidate that only moved the object: was it imagined one step further, and which
    # grasp works from where the nudge leaves the object (None: none does)?
    looked_ahead: bool = False
    leads_to: Action | None = None


def score(result: SkillResult, failure: FailureReport) -> float:
    if not result.ok:
        return -1.0
    if failure.is_failure:
        return 0.0
    return 1.0 - min(result.duration_s, 30.0) / 300.0   # prefer faster among successes


class Imagination:
    def __init__(self, world, perception: Perception, max_candidates: int = 3, capture: bool = False):
        self.world = world
        self.capture = capture   # record each imagined rollout as frames (for demo videos)
        self.perception = perception
        self.max_candidates = max_candidates

    def _physical_state(self, step: int) -> WorldState:
        """Imagined outcomes are judged on the simulator's state, not through the camera:
        an object knocked out of view can still be grasped in physics."""
        obs = self.world.observe()
        return WorldState(copy.deepcopy(obs.gt_objects), obs.gripper_holding, step)

    def evaluate(self, candidates: list[Action], goal: Goal, step: int,
                 then: Action | None = None) -> list[Imagined]:
        """Run each candidate (and `then`, if given) from the same state. A grasp that
        holds but cannot be placed should lose to one that can, so the follow-up and the
        goal check are part of the score."""
        set_recording = getattr(self.world, "set_recording", lambda on: None)
        set_recording(False)
        snap = self.world.snapshot()
        before = self._physical_state(step)
        out = []
        try:
            for a in candidates[: self.max_candidates]:
                capture = self.capture and hasattr(self.world, "push_capture")
                if capture:
                    self.world.push_capture()
                res = self.world.execute(a)
                trace = getattr(self.world, "last_trace", None)
                after = self._physical_state(step)
                fail = diagnose(a, res, before, after, goal.target)
                then_trace = None
                if then is not None and res.ok and not fail.is_failure:
                    res2 = self.world.execute(then)
                    then_trace = getattr(self.world, "last_trace", None)
                    if not res2.ok:
                        fail = FailureReport(FailureType.CONTROLLER_ERROR, f"then {then.short()}: {res2.message}")
                    elif not self.world.is_goal_satisfied(goal):
                        fail = goal_not_met(self.world, goal, f"after {a.short()} and {then.short()}")
                frames = self.world.pop_capture() if capture else []
                frame = self.world.frame()
                if frame is not None:
                    frames.append(frame)
                out.append(Imagined(a, res, fail, score(res, fail), frames, trace,
                                    then if then_trace is not None else None, then_trace))
                self.world.restore(snap)
        finally:
            self.world.restore(snap)
            set_recording(True)
        return out

    def choose(self, candidates: list[Action], goal: Goal, step: int,
               then: Action | None = None) -> tuple[Action, list[Imagined]]:
        if len(candidates) <= 1:
            return candidates[0], []
        imagined = self.evaluate(candidates, goal, step, then)
        # The planner's proposal stands if physics says it works; it is only overruled when it fails.
        if imagined[0].result.ok and not imagined[0].failure.is_failure:
            return imagined[0].action, imagined
        works = [i for i in imagined if i.result.ok and not i.failure.is_failure]
        if works:
            return max(works, key=lambda i: i.score).action, imagined
        # Nothing works. A candidate that moves the object might leave it somewhere graspable,
        # but blindly trying it rarely does: imagine one step further and nudge only when a
        # grasp works afterwards.
        movers = sorted((i for i in imagined if i.failure.type == FailureType.OBJECT_MOVED),
                        key=lambda i: i.score, reverse=True)
        for m in movers[:MAX_NUDGES_IMAGINED]:
            m.looked_ahead = True
            m.leads_to = self._grasp_after(m, candidates, goal, step, then)
            if m.leads_to is not None:
                return m.action, imagined
        best = max(imagined, key=lambda i: (i.score, i.failure.type == FailureType.OBJECT_MOVED))
        return best.action, imagined

    def _grasp_after(self, nudge: Imagined, candidates: list[Action], goal: Goal, step: int,
                     then: Action | None) -> Action | None:
        """Replay the nudge, then imagine every candidate from where it leaves the object."""
        if nudge.trace is None or not hasattr(self.world, "replay"):
            return None
        set_recording = getattr(self.world, "set_recording", lambda on: None)
        capture, self.capture = self.capture, False
        set_recording(False)
        snap = self.world.snapshot()
        try:
            self.world.replay(nudge.trace)
            for i in self.evaluate(candidates, goal, step, then):
                if i.result.ok and not i.failure.is_failure:
                    return i.action
            return None
        finally:
            self.world.restore(snap)
            set_recording(True)
            self.capture = capture
