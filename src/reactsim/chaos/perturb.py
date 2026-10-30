"""Chaos Mode: the world fights back.

Three sources of perturbation:
  * scripted  — fixed step/event pairs, for reproducible benchmarks
  * human     — anyone types "throw a ball at the banana" during a live demo; Muse Spark
                turns the sentence into an event (keywords if it is offline)
  * adversary — (stretch) a second Muse Spark prompt that tries to make the robot fail

Human sentences can also be scripted ({step: N, say: "..."}), so a demo video of the
natural-language path is reproducible.
"""

from __future__ import annotations

import json
import queue
import re

import numpy as np

from reactsim.types import CONTAINER_KINDS, PerturbEvent, WorldState

KINDS = ("knock", "teleport", "drop")
MAX_SHIFT = 0.25                       # teleport at most this far (m)
TABLE_X, TABLE_Y = (-0.25, 0.25), (-0.35, 0.35)   # keep teleported objects on the workspace

CHAOS_SPEC = """\
You turn a spectator's mischief into ONE disturbance for a robot tabletop simulation.
Coordinates: x points away from the robot, y to the robot's left, metres.
Kinds:
- "knock": a baseball is thrown at the object (even one in the gripper); dx, dy is the direction it flies.
- "teleport": the object is picked up and put down dx, dy away (|dx|, |dy| <= 0.25), as if someone moved it.
- "drop": the gripper is forced open and lets go of whatever it holds (object null).
Use only the objects listed as movable. If the sentence names none, choose the one that makes
the mischief work (e.g. the held object for "make it drop").
Answer with JSON only:
{"kind": ..., "object": <name or null>, "dx": <m>, "dy": <m>, "say": "<what happens, under 12 words>"}"""


class Chaos:
    def __init__(self, scripted: list[dict] | None = None, llm=None):
        self.scripted: dict[int, list] = {}
        for e in scripted or []:   # an event, or a sentence to parse when the step comes
            item = e["say"] if "say" in e else PerturbEvent(**e["event"])
            self.scripted.setdefault(int(e["step"]), []).append(item)
        self.llm = llm
        self.inbox: "queue.Queue[str]" = queue.Queue()   # fed by the dashboard / stdin thread
        self.log: list[PerturbEvent] = []

    def submit(self, text: str) -> None:
        self.inbox.put(text)

    def has_events(self, step: int) -> bool:
        return step in self.scripted or not self.inbox.empty()

    def events_for_step(self, step: int, state: WorldState) -> list[PerturbEvent]:
        items = list(self.scripted.get(step, []))
        while not self.inbox.empty():
            items.append(self.inbox.get_nowait())
        events = []
        for item in items:
            ev = self.parse_human(item, state) if isinstance(item, str) else item
            if ev is not None:
                events.append(ev)
        self.log.extend(events)
        return events

    def parse_human(self, text: str, state: WorldState) -> PerturbEvent | None:
        ev = None
        if self.llm is not None:
            from reactsim.planning.schema import extract_json
            try:
                d = extract_json(self.llm.chat(CHAOS_SPEC, _prompt(text, state)))
                ev = PerturbEvent(str(d.get("kind", "")), d.get("object"), float(d.get("dx") or 0),
                                  float(d.get("dy") or 0), source="human", said=text,
                                  meaning=str(d.get("say", ""))[:80])
            except Exception:
                ev = None
            ev = _validate(ev, state)
        if ev is None:
            ev = _validate(_keyword_parse(text, state), state)
        return ev


def _movable(state: WorldState) -> list[str]:
    return [n for n, o in state.objects.items() if o.kind not in CONTAINER_KINDS]


def _prompt(text: str, state: WorldState) -> str:
    objs = {n: [round(float(v), 2) for v in state.objects[n].position[:2]] for n in _movable(state)}
    return (f"Movable objects (x, y): {json.dumps(objs)}\n"
            f"The robot is holding: {state.holding or 'nothing'}\n"
            f"Spectator: {text}")


def _validate(ev: PerturbEvent | None, state: WorldState) -> PerturbEvent | None:
    """Keep only events the world can carry out; clamp teleports to the workspace."""
    if ev is None or ev.kind not in KINDS:
        return None
    if ev.kind == "drop":
        return ev if state.holding else None
    if ev.object not in _movable(state):
        return None
    if ev.kind == "knock":
        if np.hypot(ev.dx, ev.dy) < 1e-6:   # no direction given: away from the robot's reach
            ev.dx, ev.dy = 0.1, 0.0
        return ev
    p = state.objects[ev.object].position
    dx = float(np.clip(ev.dx, -MAX_SHIFT, MAX_SHIFT))
    dy = float(np.clip(ev.dy, -MAX_SHIFT, MAX_SHIFT))
    ev.dx = float(np.clip(p[0] + dx, *TABLE_X) - p[0])
    ev.dy = float(np.clip(p[1] + dy, *TABLE_Y) - p[1])
    return ev if np.hypot(ev.dx, ev.dy) >= 0.02 else None   # "do a dance": nothing actually moves


def _keyword_parse(text: str, state: WorldState) -> PerturbEvent | None:
    t = text.lower()
    if re.search(r"\b(drop|release|let go|open)\b", t):
        return PerturbEvent("drop", source="human", said=text, meaning="the gripper is forced open")
    names = _movable(state)
    obj = next((n for n in names if n.split("_")[0] in t), None)
    if obj is None and re.search(r"\b(it|that|hand|gripper)\b", t) and state.holding:
        obj = state.holding
    if obj is None:
        return None
    if re.search(r"\b(knock|hit|throw|ball|smash)\b", t):
        return PerturbEvent("knock", obj, -0.08, 0.0, source="human", said=text,
                            meaning=f"a baseball is thrown at the {obj}")
    return PerturbEvent("teleport", obj, 0.12, -0.10, source="human", said=text,
                        meaning=f"someone moves the {obj}")
