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
