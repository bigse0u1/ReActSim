"""Chaos Mode: the world fights back.

Three sources of perturbation:
  * scripted  — fixed step/event pairs, for reproducible benchmarks
  * human     — anyone types "knock the red cube off" during a live demo
  * adversary — (stretch) a second Muse Spark prompt that tries to make the robot fail
"""

from __future__ import annotations

import queue
import re

from reactsim.types import PerturbEvent

CHAOS_SPEC = """\
Convert the user's mischief into ONE JSON perturbation for a tabletop sim:
{"kind": "teleport" | "knock" | "drop", "object": <object name or null>, "dx": <m>, "dy": <m>}
Keep |dx|,|dy| <= 0.2. "drop" makes the gripper release whatever it holds."""


class Chaos:
    def __init__(self, scripted: list[dict] | None = None, llm=None):
        self.scripted = {int(e["step"]): PerturbEvent(**e["event"]) for e in (scripted or [])}
        self.llm = llm
        self.inbox: "queue.Queue[str]" = queue.Queue()   # fed by the dashboard / stdin thread
        self.log: list[PerturbEvent] = []

    def submit(self, text: str) -> None:
        self.inbox.put(text)

    def events_for_step(self, step: int, object_names: list[str]) -> list[PerturbEvent]:
        events = []
        if step in self.scripted:
            events.append(self.scripted[step])
        while not self.inbox.empty():
            ev = self.parse_human(self.inbox.get_nowait(), object_names)
            if ev:
                events.append(ev)
        self.log.extend(events)
        return events

    def parse_human(self, text: str, object_names: list[str]) -> PerturbEvent | None:
        if self.llm is not None:
            import json

            from reactsim.planning.schema import extract_json
            try:
                d = extract_json(self.llm.chat(CHAOS_SPEC, f"Objects: {json.dumps(object_names)}\nUser: {text}"))
                return PerturbEvent(d["kind"], d.get("object"), float(d.get("dx", 0)),
                                    float(d.get("dy", 0)), source="human")
            except Exception:
                pass
        return _keyword_parse(text, object_names)


def _keyword_parse(text: str, object_names: list[str]) -> PerturbEvent | None:
    t = text.lower()
    if re.search(r"\b(drop|release|let go)\b", t):
        return PerturbEvent("drop", source="human")
    obj = next((n for n in object_names if n.split("_")[0] in t), None)
    if obj is None:
        return None
    return PerturbEvent("knock" if "knock" in t or "hit" in t else "teleport",
                        obj, 0.12, -0.10, source="human")
