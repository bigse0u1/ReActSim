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
