"""Offline stand-in for Muse Spark so the LLM code path (prompt -> JSON -> validator)
runs without an API key. Answers are templated from the world state in the prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from reactsim.types import CONTAINER_KINDS, GRASP_MODES


@dataclass
class _Call:
    latency_s: float = 0.0


class MockMuseClient:
    def __init__(self):
        self.calls: list[_Call] = []

    def chat(self, system: str, user: str) -> str:
        self.calls.append(_Call())
        state = json.loads(re.search(r"state: (\{.*\})", user).group(1))
        objs = {o["name"]: o for o in state["objects"]}
        text = user.lower()
        m = re.search(r"instruction: (.*)", text)
        instruction = m.group(1) if m else text   # the state JSON names every object; match the instruction only
        goals = _mock_goals(text, instruction, objs)
        target, plate = goals[0]["target"], goals[0]["reference"]
        grasp = "top"
        rationale = f"Pick {target} and place it on {plate}."
        if "change strategy" in text and "failure: grasp_failed" not in text:
            grasp = GRASP_MODES[(text.count("same failure") + 1) % len(GRASP_MODES)]
            rationale = f"The same recovery failed again; switching to a {grasp} grasp."
        elif "failure: grasp_failed" in text:
            m = re.search(r'"grasp": "(\w+)"', user)
            failed = m.group(1) if m else "top"
            grasp = GRASP_MODES[(GRASP_MODES.index(failed) + 1) % len(GRASP_MODES)]
            rationale = f"The {m.group(1) if m else 'previous'} grasp slipped; try a {grasp} grasp."
        elif "failure: object_moved" in text:
            rationale = f"{target} was moved; re-plan from its new position."
        actions, holding = [], state["holding"]
        for k, g in enumerate(goals):
            if holding != g["target"]:
                actions.append({"type": "pick", "target": g["target"], "grasp": grasp if k == 0 else "top"})
            actions.append({"type": "place_on", "reference": g["reference"]})
            holding = None
        return json.dumps({
            "goals": goals,
            "rationale": rationale,
            "actions": actions,
            "alternatives": [{"type": "pick", "target": target, "grasp": g}
                             for g in GRASP_MODES if g != grasp],
        })


def _mock_goals(text: str, instruction: str, objs: dict) -> list[dict]:
    """On a replan, the goals still to do are listed in the prompt; otherwise every object named
    in the instruction goes on/in the container named after it (default: the first one)."""
    m = re.search(r"goals still to do: (.*)", text)
    if m:
        out = []
        for part in m.group(1).split(";"):
            words = part.split()
            if len(words) == 3 and words[0] in objs and words[2] in objs:
                out.append({"target": words[0], "relation": words[1], "reference": words[2]})
        if out:
            return out
    containers = [n for n, o in objs.items() if o["kind"] in CONTAINER_KINDS]

    def pos(n):
        hits = [i for i in (instruction.find(n.replace("_", " ")), instruction.find(objs[n]["kind"])) if i >= 0]
        return min(hits) if hits else -1

    named = sorted((pos(n), n) for n, o in objs.items() if o["kind"] not in CONTAINER_KINDS and pos(n) >= 0)
    if not named:   # fall back to the first movable object
        named = [(0, next(n for n, o in objs.items() if o["kind"] not in CONTAINER_KINDS))]
    goals = []
    for p, n in named:
        after = sorted((pos(c), c) for c in containers if pos(c) > p)
        ref = after[0][1] if after else containers[0]
        goals.append({"target": n, "relation": "in" if objs[ref]["kind"] == "bowl" else "on", "reference": ref})
    return goals
