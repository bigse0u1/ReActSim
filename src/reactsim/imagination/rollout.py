"""Imagination: try candidate actions in a forked copy of the physics, then commit to the best.

This is real lookahead (the simulator steps the candidate forward and is restored),
not a heuristic feasibility score. On a real robot the same idea needs a
digital twin; in the demo we say exactly that.

Note: if the world is wrapped in RecordEpisode, imagined rollouts are recorded
too. Either pause recording or use those frames as the "ghost" split-screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reactsim.evaluation.diagnosis import diagnose
from reactsim.perception.grounding import Perception
from reactsim.types import Action, FailureReport, Goal, SkillResult


@dataclass
class Imagined:
    action: Action
    result: SkillResult
    failure: FailureReport
    score: float
    frames: list = field(default_factory=list)


def score(result: SkillResult, failure: FailureReport) -> float:
    if not result.ok:
        return -1.0
    if failure.is_failure:
        return 0.0
    return 1.0 - min(result.duration_s, 30.0) / 300.0   # prefer faster among successes


class Imagination:
    def __init__(self, world, perception: Perception, max_candidates: int = 3):
        self.world = world
        self.perception = perception
        self.max_candidates = max_candidates

    def evaluate(self, candidates: list[Action], goal: Goal, step: int) -> list[Imagined]:
        set_recording = getattr(self.world, "set_recording", lambda on: None)
        set_recording(False)
        snap = self.world.snapshot()
        before = self.perception.perceive(self.world.observe(), step)
        out = []
        try:
            for a in candidates[: self.max_candidates]:
                res = self.world.execute(a)
                after = self.perception.perceive(self.world.observe(), step)
                fail = diagnose(a, res, before, after, goal.target)
                frame = self.world.frame()
                out.append(Imagined(a, res, fail, score(res, fail), [frame] if frame is not None else []))
                self.world.restore(snap)
        finally:
            self.world.restore(snap)
            set_recording(True)
        return out

    def choose(self, candidates: list[Action], goal: Goal, step: int) -> tuple[Action, list[Imagined]]:
        if len(candidates) <= 1:
            return candidates[0], []
        imagined = self.evaluate(candidates, goal, step)
        best = max(imagined, key=lambda i: i.score)
        return best.action, imagined
