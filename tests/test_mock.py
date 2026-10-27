"""The offline planner must take the target from the instruction, not from the state JSON
(which names every object)."""

import json

from reactsim.planning.mock import MockMuseClient

STATE = {"step": 0, "holding": None, "objects": [
    {"name": n, "kind": n, "color": "", "position": [0, 0, 0], "held": False}
    for n in ("orange", "can", "cup", "plate")]}


def test_target_comes_from_instruction():
    user = f"Instruction: Put the cup on the plate\nWorld state: {json.dumps(STATE)}"
    plan = json.loads(MockMuseClient().chat("", user))
    assert plan["actions"][0]["target"] == "cup"
