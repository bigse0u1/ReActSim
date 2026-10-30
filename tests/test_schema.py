import numpy as np
import pytest

from reactsim.planning.schema import PlanValidationError, parse_plan
from reactsim.types import ActionType, ObjectState, WorldState


def _state(holding=None):
    objs = {
        "red_cube": ObjectState("red_cube", np.zeros(3), "red", "cube"),
        "plate": ObjectState("plate", np.ones(3), "white", "plate"),
    }
    return WorldState(objs, holding, 0)


GOOD = '''```json
{"goal": {"target": "red_cube", "relation": "on", "reference": "plate"},
 "rationale": "ok",
 "actions": [{"type": "pick", "target": "red_cube"}, {"type": "place_on", "reference": "plate"}],
 "alternatives": [{"type": "pick", "target": "red_cube", "grasp": "side"},
                  {"type": "pick", "target": "red_cube", "grasp": "sideways"},
                  {"type": "pick", "target": "ghost"}]}
```'''


def test_parse_fenced_plan_and_drop_bad_alternative():
    plan, alts = parse_plan(GOOD, _state())
    assert [a.type for a in plan.actions] == [ActionType.PICK, ActionType.PLACE_ON]
    assert len(alts) == 1 and alts[0].grasp == "side"


@pytest.mark.parametrize("bad", [
    "not json",
    '{"goal": {"target": "red_cube", "reference": "plate"}, "actions": [{"type": "fly"}]}',
    '{"goal": {"target": "red_cube", "reference": "plate"}, "actions": [{"type": "pick", "target": "ghost"}]}',
    '{"goal": {"target": "red_cube", "reference": "plate"}, "actions": [{"type": "place_on", "reference": "plate"}]}',
    '{"goal": {"target": "red_cube", "reference": "plate"}, "actions": [{"type": "pick", "target": "plate"}]}',
])
def test_rejects_invalid(bad):
    with pytest.raises(PlanValidationError):
        parse_plan(bad, _state())


def _tidy_state():
    from reactsim.types import ObjectState, WorldState
    objs = {n: ObjectState(n, np.zeros(3), kind=k) for n, k in
            (("banana", "banana"), ("orange", "orange"), ("can", "can"), ("plate", "plate"), ("bowl", "bowl"))}
    return WorldState(objs, None, 0)


def test_plan_with_several_goals():
    import json
    from reactsim.planning.schema import parse_plan
    text = json.dumps({"goals": [{"target": "banana", "relation": "on", "reference": "plate"},
                                 {"target": "can", "relation": "in", "reference": "bowl"}],
                       "actions": [{"type": "pick", "target": "banana"}, {"type": "place_on", "reference": "plate"},
                                   {"type": "pick", "target": "can"}, {"type": "place_on", "reference": "bowl"}]})
    plan, _ = parse_plan(text, _tidy_state())
    assert [(g.target, g.reference) for g in plan.goals] == [("banana", "plate"), ("can", "bowl")]


def test_goal_must_name_a_container():
    import json
    import pytest
    from reactsim.planning.schema import PlanValidationError, parse_plan
    text = json.dumps({"goals": [{"target": "banana", "relation": "on", "reference": "can"}],
                       "actions": [{"type": "pick", "target": "banana"}, {"type": "place_on", "reference": "can"}]})
    with pytest.raises(PlanValidationError, match="not a container"):
        parse_plan(text, _tidy_state())


def test_keyword_baseline_reads_several_goals():
    from reactsim.planning.planner import _parse_goal_keywords
    goals = _parse_goal_keywords("Put the banana and the orange on the plate, and the can in the bowl", _tidy_state())
    assert [(g.target, g.relation, g.reference) for g in goals] == [
        ("banana", "on", "plate"), ("orange", "on", "plate"), ("can", "in", "bowl")]
    # no language understanding: "the fruit" names nothing it knows
    assert [g.target for g in _parse_goal_keywords("Put the fruit on the plate and the can in the bowl",
                                                   _tidy_state())] == ["can"]
