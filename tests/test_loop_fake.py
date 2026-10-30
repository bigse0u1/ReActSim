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


def test_no_working_grasp_is_reported_not_attempted():
    """If every imagined grasp fails (here: out of reach), the agent does not execute one;
    it reports why, and stops after MAX_NO_WORKING_GRASP repeats."""
    import numpy as np

    from reactsim.agent import MAX_NO_WORKING_GRASP, ReActAgent
    from reactsim.imagination.rollout import Imagination
    from reactsim.perception.grounding import Perception
    from reactsim.planning.mock import MockMuseClient
    from reactsim.planning.planner import MusePlanner
    from reactsim.types import ActionType
    from reactsim.worlds.fake import FakeWorld

    class FarWorld(FakeWorld):
        executed = []

        def reset(self, seed):
            super().reset(seed)
            self.state.objects["banana"].position = np.array([1.5, 0.0, 0.02])   # beyond reach

        def execute(self, action):
            if action.type == ActionType.PICK and not getattr(self, "_imagining", False):
                FarWorld.executed.append(action.short())
            return super().execute(action)

    world = FarWorld()
    perception = Perception(None)
    imagination = Imagination(world, perception, max_candidates=3)
    orig = imagination.evaluate

    def evaluate(*a, **k):
        world._imagining = True
        try:
            return orig(*a, **k)
        finally:
            world._imagining = False
    imagination.evaluate = evaluate
    agent = ReActAgent(world, perception, MusePlanner(MockMuseClient()), imagination=imagination,
                       max_steps=30, max_replans=10)
    rec = agent.run("Put the banana on the plate", seed=0)
    assert not rec.success
    assert rec.stop_reason == f"gave up: no_working_grasp repeated {MAX_NO_WORKING_GRASP} times"
    assert FarWorld.executed == []                     # never tried a pick it knew would fail
    assert "no grasp works on banana" in rec.steps[0]["detail"]


def test_several_goals_all_have_to_hold():
    cfg = copy.deepcopy(CFG)
    cfg["world"] = "fake"
    cfg["planning"]["planner"] = "muse"
    cfg["chaos"]["enabled"] = False
    agent = build_agent(cfg, offline=True)
    rec = agent.run("Put the banana and the can on the plate", seed=0)
    assert rec.success
    assert rec.goals == ["banana on plate", "can on plate"]
    assert sum(s["action"].startswith("place_on") and s["failure"] == "none" for s in rec.steps) == 2


def test_imagination_prefers_a_working_grasp_over_a_nudge():
    """A nudge is only imagined further when no candidate works; when one is found to make a
    grasp work afterwards it is chosen, otherwise it is marked as not helping."""
    from reactsim.imagination.rollout import Imagination, Imagined
    from reactsim.types import Action, ActionType, FailureReport, FailureType, Goal, SkillResult

    def im(grasp, kind):
        fail = FailureReport(FailureType(kind))
        return Imagined(Action(ActionType.PICK, "can", grasp=grasp), SkillResult(True), fail,
                        1.0 if kind == "none" else 0.0, trace=([], SkillResult(True)))

    goal = Goal("can", "on", "plate")
    imagination = Imagination(world=None, perception=None, max_candidates=3)
    looked = []
    imagination._grasp_after = lambda m, *a: looked.append(m) or after
    after = Action(ActionType.PICK, "can", grasp="top")

    imagination.evaluate = lambda *a: [im("side", "object_moved"), im("top", "none")]
    action, _ = imagination.choose([None, None], goal, 0)
    assert action.grasp == "top" and looked == []          # the working grasp, no lookahead

    imagination.evaluate = lambda *a: [im("side", "object_moved"), im("top", "grasp_failed")]
    action, imagined = imagination.choose([None, None], goal, 0)
    assert action.grasp == "side" and imagined[0].leads_to is after

    after = None
    action, imagined = imagination.choose([None, None], goal, 0)
    assert imagined[0].looked_ahead and imagined[0].leads_to is None
