"""Offline stand-in for Muse Spark so the LLM code path (prompt -> JSON -> validator)
runs without an API key. Answers are templated from the world state in the prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from reactsim.types import GRASP_MODES


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
        target = next((n for n, o in objs.items()
                       if o["kind"] != "plate" and (n.replace("_", " ") in text or o["kind"] in text)),
                      next(n for n, o in objs.items() if o["kind"] != "plate"))
        plate = next(n for n, o in objs.items() if o["kind"] == "plate")
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
        actions = []
        if state["holding"] != target:
            actions.append({"type": "pick", "target": target, "grasp": grasp})
        actions.append({"type": "place_on", "reference": plate})
        return json.dumps({
            "goal": {"target": target, "relation": "on", "reference": plate},
            "rationale": rationale,
            "actions": actions,
            "alternatives": [{"type": "pick", "target": target, "grasp": g}
                             for g in GRASP_MODES if g != grasp],
        })
