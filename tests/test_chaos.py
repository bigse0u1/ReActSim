import json

import numpy as np

from reactsim.chaos.perturb import Chaos
from reactsim.types import ObjectState, PerturbEvent, WorldState


def _state(holding=None):
    objs = {"banana": ObjectState("banana", np.array([0.05, -0.06, 0.02]), kind="banana"),
            "mug": ObjectState("mug", np.array([-0.10, -0.14, 0.04]), kind="mug"),
            "plate": ObjectState("plate", np.array([0.0, 0.17, 0.01]), kind="plate")}
    return WorldState(objs, holding, 0)


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def chat(self, system, user):
        self.seen.append(user)
        return json.dumps(self.reply) if isinstance(self.reply, dict) else self.reply


def test_muse_reads_the_sentence_and_sees_positions_and_the_hand():
    llm = FakeLLM({"kind": "knock", "object": "banana", "dx": 0, "dy": 0.2, "say": "a ball hits the banana"})
    ev = Chaos(llm=llm).parse_human("make it drop the fruit", _state(holding="banana"))
    assert (ev.kind, ev.object, ev.dy, ev.source) == ("knock", "banana", 0.2, "human")
    assert ev.said == "make it drop the fruit" and ev.meaning == "a ball hits the banana"
    assert "holding: banana" in llm.seen[0] and '"mug": [-0.1, -0.14]' in llm.seen[0]
    assert "plate" not in llm.seen[0]                  # containers are not offered


def test_invalid_answers_are_rejected_or_clamped():
    # A container is not movable: fall back to the keyword parser (which finds the banana).
    ev = Chaos(llm=FakeLLM({"kind": "teleport", "object": "plate", "dx": 0.1})).parse_human(
        "move the banana", _state())
    assert (ev.kind, ev.object) == ("teleport", "banana")
    # Teleports stay within reach and on the workspace.
    ev = Chaos(llm=FakeLLM({"kind": "teleport", "object": "mug", "dx": -3.0, "dy": 0.0})).parse_human(
        "throw the mug off the table", _state())
    assert ev.dx == -0.15                              # x clamped to the workspace edge (-0.25)
    # Not JSON at all, and nothing the keywords recognise: no event.
    assert Chaos(llm=FakeLLM("sorry, I cannot")).parse_human("do a dance", _state()) is None
    # A "move" that moves nothing is not an event.
    assert Chaos(llm=FakeLLM({"kind": "teleport", "object": "mug", "dx": 0, "dy": 0})).parse_human(
        "do a little dance", _state()) is None
    # "drop" with an empty hand changes nothing.
    assert Chaos(llm=FakeLLM({"kind": "drop", "object": None})).parse_human("let go", _state()) is None


def test_keywords_offline_and_scripted_sentences():
    chaos = Chaos([{"step": 1, "say": "throw a ball at it"}, {"step": 1, "event": {"kind": "drop"}}])
    assert not chaos.has_events(0) and chaos.has_events(1)
    evs = chaos.events_for_step(1, _state(holding="banana"))
    assert [e.kind for e in evs] == ["knock", "drop"]
    assert evs[0].object == "banana" and evs[0].said == "throw a ball at it"   # "it" = what it holds
    assert isinstance(evs[1], PerturbEvent) and evs[1].source == "scripted"
