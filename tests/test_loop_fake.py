import copy
from pathlib import Path

from reactsim.build import build_agent, load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")


def _run(planner, imagine=True, chaos=True):
    cfg = copy.deepcopy(CFG)
    cfg["world"] = "fake"
    cfg["planning"]["planner"] = planner
    cfg["imagination"]["enabled"] = imagine
    cfg["chaos"]["enabled"] = chaos
    agent = build_agent(cfg, offline=True)
    return agent.run("Put the banana on the plate", seed=0)


def test_fixed_baseline_fails_on_slippery_grasp():
    rec = _run("fixed", chaos=False)
    assert not rec.success and rec.initial_failure


def test_rule_baseline_recovers_by_retrying():
    rec = _run("rule", chaos=False)
    assert rec.success and rec.recovery_attempts >= 1


def test_imagination_avoids_slippery_grasp():
    rec = _run("muse", imagine=True, chaos=False)
    assert rec.success and rec.recovery_attempts == 0
    assert "[top]" not in rec.steps[0]["action"]


def test_muse_recovers_from_knock():
    rec = _run("muse", imagine=True, chaos=True)
    assert rec.success and rec.perturbations == 1
    assert any(s["failure"] == "object_moved" for s in rec.steps)


class _UnreachableGoalWorld:
    """FakeWorld whose success check never passes: the plate 'moves away' after every place."""

    def __new__(cls):
        from reactsim.worlds.fake import FakeWorld

        class W(FakeWorld):
            def goal_report(self, goal):
                r = super().goal_report(goal)
                r["over_reference"] = {"ok": False, "value_cm": 20.0}
                return r
        return W()


def test_goal_not_met_is_explained_and_repeats_stop():
    from reactsim.agent import MAX_SAME_FAILURE, ReActAgent
    from reactsim.perception.grounding import Perception
    from reactsim.planning.mock import MockMuseClient
    from reactsim.planning.planner import MusePlanner

    agent = ReActAgent(_UnreachableGoalWorld(), Perception(None), MusePlanner(MockMuseClient()),
                       max_steps=30, max_replans=10)
    rec = agent.run("Put the banana on the plate", seed=0)
    assert not rec.success
    assert rec.stop_reason.startswith("gave up: goal_not_met")
    assert str(MAX_SAME_FAILURE) in rec.stop_reason
    assert rec.recovery_attempts == MAX_SAME_FAILURE - 1   # stopped instead of burning the replan budget
