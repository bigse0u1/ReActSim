"""The JSON contract between Muse Spark and the robot, plus the action validator.

The LLM proposes; this module disposes. Anything that does not parse, names an
unknown object, or violates a precondition is rejected before it reaches the arm.
"""

from __future__ import annotations

import json
import re
from typing import Any

from reactsim.types import Action, ActionType, Goal, Plan, WorldState

ACTION_SPEC = """\
Allowed actions (JSON objects):
  {"type": "pick", "target": <object>, "grasp": "top" | "side"}
  {"type": "place_on", "reference": <object>}          # places what is held
  {"type": "push_aside", "target": <object>}           # clear an obstacle
  {"type": "reobserve"}                                # look again before acting
  {"type": "home"}
  {"type": "done"}
  {"type": "give_up"}
Respond with ONLY a JSON object:
{
  "goal": {"target": <object>, "relation": "on", "reference": <object>},
  "rationale": <one or two sentences>,
  "actions": [<action>, ...],
  "alternatives": [<action>, ...]   # optional: other options for the FIRST action
}"""


class PlanValidationError(ValueError):
    pass


def extract_json(text: str) -> dict[str, Any]:
    """Accept bare JSON or JSON inside a ```json fence."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlanValidationError(f"invalid JSON: {e}") from e


def parse_action(d: dict[str, Any]) -> Action:
    try:
        t = ActionType(d["type"])
    except (KeyError, ValueError) as e:
        raise PlanValidationError(f"unknown action type: {d.get('type')!r}") from e
    grasp = d.get("grasp", "top")
    if grasp not in ("top", "side"):
        raise PlanValidationError(f"unknown grasp {grasp!r}")
    return Action(t, d.get("target"), d.get("reference"), grasp, d.get("rationale", ""))


def validate_actions(actions: list[Action], state: WorldState) -> None:
    """Static feasibility: object ids exist and preconditions chain correctly."""
    names = set(state.objects)
    holding = state.holding
    for i, a in enumerate(actions):
        for field in ("target", "reference"):
            v = getattr(a, field)
            if v is not None and v not in names:
                raise PlanValidationError(f"action {i}: unknown object {v!r}")
        if a.type == ActionType.PICK:
            if a.target is None:
                raise PlanValidationError(f"action {i}: pick needs a target")
            if state.objects[a.target].kind == "plate":
                raise PlanValidationError(f"action {i}: cannot pick the plate")
            if holding is not None:
                raise PlanValidationError(f"action {i}: pick while holding {holding}")
            holding = a.target
        elif a.type == ActionType.PLACE_ON:
            if a.reference is None:
                raise PlanValidationError(f"action {i}: place_on needs a reference")
            if holding is None:
                raise PlanValidationError(f"action {i}: place_on with empty gripper")
            holding = None
        elif a.type == ActionType.PUSH_ASIDE and a.target is None:
            raise PlanValidationError(f"action {i}: push_aside needs a target")


def parse_plan(text: str, state: WorldState) -> tuple[Plan, list[Action]]:
    d = extract_json(text)
    try:
        g = d["goal"]
        goal = Goal(g["target"], g.get("relation", "on"), g["reference"])
    except (KeyError, TypeError) as e:
        raise PlanValidationError(f"missing goal: {e}") from e
    if goal.target not in state.objects or goal.reference not in state.objects:
        raise PlanValidationError(f"goal names unknown objects: {goal}")
    actions = [parse_action(a) for a in d.get("actions", [])]
    if not actions:
        raise PlanValidationError("empty plan")
    validate_actions(actions, state)
    alternatives = []
    for a in d.get("alternatives", []) or []:
        try:
            alt = parse_action(a)
            validate_actions([alt], state)
            alternatives.append(alt)
        except PlanValidationError:
            continue   # a bad alternative is dropped, not fatal
    return Plan(goal, actions, d.get("rationale", "")), alternatives
