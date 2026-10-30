"""Planners: Muse Spark (proposed), rule-based retry (baseline B), fixed script (baseline A)."""

from __future__ import annotations

import json
from typing import Protocol

from reactsim.planning.schema import ACTION_SPEC, PlanValidationError, parse_plan
from reactsim.types import CONTAINER_KINDS, GRASP_MODES, Action, ActionType, FailureReport, FailureType, Goal, Plan, WorldState

SYSTEM_PROMPT = f"""\
You are the high-level planner of a robot arm on a tabletop.
An instruction may ask for several objects to be moved: list every one as a goal, decide a
sensible order, and plan all the picks and places. Ground words like "the fruit" to the
objects in the world state that fit them.
You never control joints; you choose skills from a fixed list. Low-level motion
planning, grasping and the success check are done by the system.
When a failure report is given, explain the likely physical cause in the
rationale and choose a recovery that addresses that cause (e.g. a different grasp,
re-observing, clearing an obstacle), not a blind repeat.
A goal_not_met report lists each success condition the physics check measured; read
the values before deciding the task was not done. If a failure is marked as repeated,
the last recovery did not work: do something different, or give_up and say why.
A "no_working_grasp" report means Imagination already tried every grasp in physics: none held
the object and none even moved it. Retrying a grasp will not help unless something changes
(e.g. push an obstacle aside); otherwise give_up and explain why.

{ACTION_SPEC}"""


class Planner(Protocol):
    def plan(self, instruction: str, state: WorldState,
             history: list[dict]) -> tuple[Plan, list[Action]]: ...

    def replan(self, instruction: str, goals: list[Goal], state: WorldState,
               history: list[dict], failure: FailureReport,
               achieved: list[Goal] = ()) -> tuple[Plan, list[Action]]:
        """goals: the ones still to do; achieved: the ones that already hold."""
        ...


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

    def replan(self, instruction, goals, state, history, failure, achieved=()):
        user = (f"Instruction: {instruction}\n"
                f"Goals still to do: {_goals_text(goals)}\n"
                f"Goals already achieved (do not undo them): {_goals_text(achieved) or 'none'}\n"
                f"Executed so far: {json.dumps(history[-6:])}\n"
                f"FAILURE: {failure.type.value} — {failure.detail}\n"
                f"Evidence: {json.dumps(failure.evidence, default=float)}\n"
                f"Current world state: {json.dumps(state.to_prompt_dict())}\n"
                "Produce a recovery plan from the current state. List only the goals still to do.")
        return self._ask(user, state)


def _goals_text(goals) -> str:
    return "; ".join(f"{g.target} {g.relation} {g.reference}" for g in goals)


def _parse_goal_keywords(instruction: str, state: WorldState) -> list[Goal]:
    """Baseline goal parsing: every object named in the instruction (by name or kind), in the
    order mentioned, goes on/in the first container named after it (default: the first one).
    No language understanding: "the fruit" grounds to nothing."""
    text = instruction.lower()
    containers = [o for o in state.objects.values() if o.kind in CONTAINER_KINDS]
    if not containers:
        raise PlanValidationError("no container in the scene")

    def first_mention(o):
        hits = [i for i in (text.find(o.name.replace("_", " ")), text.find(o.kind)) if i >= 0]
        return min(hits) if hits else -1

    mentioned = sorted((first_mention(o), o) for o in state.objects.values()
                       if o.kind not in CONTAINER_KINDS and first_mention(o) >= 0)
    goals = []
    for pos, o in mentioned:
        after = [(first_mention(c), c) for c in containers if first_mention(c) > pos]
        ref = min(after, key=lambda x: x[0])[1] if after else containers[0]
        goals.append(Goal(o.name, "in" if ref.kind == "bowl" else "on", ref.name))
    if not goals:
        raise PlanValidationError(f"keyword parser could not ground: {instruction!r}")
    return goals


def _pick_place(goals: list[Goal], state: WorldState, grasp="top") -> list[Action]:
    acts, holding = [], state.holding
    for goal in goals:
        if holding != goal.target:
            if holding is not None:   # put down what is held first: on its own goal if it has one
                own = next((g for g in goals if g.target == holding), goal)
                acts.append(Action(ActionType.PLACE_ON, reference=own.reference))
            acts.append(Action(ActionType.PICK, goal.target, grasp=grasp))
        acts.append(Action(ActionType.PLACE_ON, reference=goal.reference))
        holding = None
    return acts


class FixedPlanner:
    """Baseline A: one scripted pick-and-place, no recovery."""

    def plan(self, instruction, state, history):
        goals = _parse_goal_keywords(instruction, state)
        return Plan(goals, _pick_place(goals, state), "fixed script"), []

    def replan(self, instruction, goals, state, history, failure, achieved=()):
        return Plan(goals, [Action(ActionType.GIVE_UP)], "no recovery"), []


class RuleBasedPlanner(FixedPlanner):
    """Baseline B: re-observe and retry the same skill sequence, cycling grasp modes on grasp failure."""

    def __init__(self):
        self.retries = 0
        self.grasp_failures = 0

    def plan(self, instruction, state, history):
        self.retries = self.grasp_failures = 0
        return super().plan(instruction, state, history)

    def replan(self, instruction, goals, state, history, failure, achieved=()):
        self.retries += 1
        self.grasp_failures += failure.type == FailureType.GRASP_FAILED
        grasp = GRASP_MODES[self.grasp_failures % len(GRASP_MODES)]
        return Plan(goals, _pick_place(goals, state, grasp), f"rule retry #{self.retries}"), []
