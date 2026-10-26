"""Planners: Muse Spark (proposed), rule-based retry (baseline B), fixed script (baseline A)."""

from __future__ import annotations

import json
from typing import Protocol

from reactsim.planning.schema import ACTION_SPEC, PlanValidationError, parse_plan
from reactsim.types import GRASP_MODES, Action, ActionType, FailureReport, FailureType, Goal, Plan, WorldState

SYSTEM_PROMPT = f"""\
You are the high-level planner of a Franka Panda robot on a tabletop.
You never control joints; you choose skills from a fixed list. Low-level motion
planning, grasping and the success check are done by the system.
When a failure report is given, explain the likely physical cause in the
rationale and choose a recovery that addresses that cause (e.g. a different grasp,
re-observing, clearing an obstacle), not a blind repeat.

{ACTION_SPEC}"""


class Planner(Protocol):
    def plan(self, instruction: str, state: WorldState,
             history: list[dict]) -> tuple[Plan, list[Action]]: ...

    def replan(self, instruction: str, goal: Goal, state: WorldState,
               history: list[dict], failure: FailureReport) -> tuple[Plan, list[Action]]: ...


class MusePlanner:
    def __init__(self, client, max_invalid_retries: int = 2):
        self.client = client
        self.max_invalid_retries = max_invalid_retries
        self.invalid_outputs = 0

    def _ask(self, user: str, state: WorldState) -> tuple[Plan, list[Action]]:
        err = None
        for _ in range(self.max_invalid_retries + 1):
            msg = user if err is None else f"{user}\n\nYour previous answer was rejected: {err}. Fix it."
            text = self.client.chat(SYSTEM_PROMPT, msg)
            try:
                return parse_plan(text, state)
            except PlanValidationError as e:
                self.invalid_outputs += 1
                err = str(e)
        raise PlanValidationError(f"planner gave no valid plan: {err}")

    def plan(self, instruction, state, history):
        user = (f"Instruction: {instruction}\n"
                f"World state: {json.dumps(state.to_prompt_dict())}")
        return self._ask(user, state)

    def replan(self, instruction, goal, state, history, failure):
        user = (f"Instruction: {instruction}\n"
                f"Goal: {goal.target} {goal.relation} {goal.reference}\n"
                f"Executed so far: {json.dumps(history[-6:])}\n"
                f"FAILURE: {failure.type.value} — {failure.detail}\n"
                f"Evidence: {json.dumps(failure.evidence, default=float)}\n"
                f"Current world state: {json.dumps(state.to_prompt_dict())}\n"
                "Produce a recovery plan from the current state.")
        return self._ask(user, state)


def _parse_goal_keywords(instruction: str, state: WorldState) -> Goal:
    """Baseline goal parsing: first non-plate object named in the instruction; plate as reference."""
    text = instruction.lower()
    target = next((o.name for o in state.objects.values()
                   if o.kind != "plate" and (o.name.replace("_", " ") in text or o.kind in text)), None)
    reference = next((o.name for o in state.objects.values() if o.kind == "plate"), None)
    if target is None or reference is None:
        raise PlanValidationError(f"keyword parser could not ground: {instruction!r}")
    return Goal(target, "on", reference)


def _pick_place(goal: Goal, state: WorldState, grasp="top") -> list[Action]:
    acts = []
    if state.holding != goal.target:
        if state.holding is not None:
            acts.append(Action(ActionType.PLACE_ON, reference=goal.reference))
        acts.append(Action(ActionType.PICK, goal.target, grasp=grasp))
    acts.append(Action(ActionType.PLACE_ON, reference=goal.reference))
    return acts


class FixedPlanner:
    """Baseline A: one scripted pick-and-place, no recovery."""

    def plan(self, instruction, state, history):
        goal = _parse_goal_keywords(instruction, state)
        return Plan(goal, _pick_place(goal, state), "fixed script"), []

    def replan(self, instruction, goal, state, history, failure):
        return Plan(goal, [Action(ActionType.GIVE_UP)], "no recovery"), []


class RuleBasedPlanner(FixedPlanner):
    """Baseline B: re-observe and retry the same skill sequence, cycling grasp modes on grasp failure."""

    def __init__(self):
        self.retries = 0
        self.grasp_failures = 0

    def plan(self, instruction, state, history):
        self.retries = self.grasp_failures = 0
        return super().plan(instruction, state, history)

    def replan(self, instruction, goal, state, history, failure):
        self.retries += 1
        self.grasp_failures += failure.type == FailureType.GRASP_FAILED
        grasp = GRASP_MODES[self.grasp_failures % len(GRASP_MODES)]
        return Plan(goal, _pick_place(goal, state, grasp), f"rule retry #{self.retries}"), []
