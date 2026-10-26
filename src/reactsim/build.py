"""Wire components together from a config dict (configs/default.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from reactsim.agent import ReActAgent
from reactsim.chaos.perturb import Chaos
from reactsim.imagination.rollout import Imagination
from reactsim.perception.grounding import Perception
from reactsim.planning.planner import FixedPlanner, MusePlanner, RuleBasedPlanner
from reactsim.viz.dashboard import Dashboard


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_world(cfg: dict, out_dir: str | None = None):
    if cfg["world"] == "fake":
        from reactsim.worlds.fake import FakeWorld
        return FakeWorld(slippery={"red_cube": {"top"}})   # forces one grasp failure to recover from
    from reactsim.worlds.maniskill import ManiSkillWorld
    m = cfg["maniskill"]
    return ManiSkillWorld(m["env_id"], m["render_mode"], m["sim_backend"],
                          record_dir=out_dir if m.get("record_video") else None)


def build_planner(cfg: dict, offline: bool = False):
    p = cfg["planning"]
    if p["planner"] == "fixed":
        return FixedPlanner()
    if p["planner"] == "rule":
        return RuleBasedPlanner()
    if offline:
        from reactsim.planning.mock import MockMuseClient
        client = MockMuseClient()
    else:
        from reactsim.planning.llm import MetaChatClient
        client = MetaChatClient(p["model"], temperature=p["temperature"])
    return MusePlanner(client, p["max_invalid_retries"])


def build_perception(cfg: dict, world) -> Perception:
    seg = cfg["perception"]["segmenter"]
    if cfg["world"] == "fake":
        return Perception(None)
    if seg == "oracle":
        from reactsim.perception.segmenter import OracleSegmenter
        world.reset(cfg["seed"])
        return Perception(OracleSegmenter(world.segmentation_ids()))
    from reactsim.perception.segmenter import Sam3Segmenter
    return Perception(Sam3Segmenter(cfg["perception"]["sam_model"]))


def build_agent(cfg: dict, offline: bool = False, out_dir: str | None = None) -> ReActAgent:
    world = build_world(cfg, out_dir)
    perception = build_perception(cfg, world)
    planner = build_planner(cfg, offline)
    is_muse = cfg["planning"]["planner"] == "muse"
    imagination = (Imagination(world, perception, cfg["imagination"]["max_candidates"])
                   if cfg["imagination"]["enabled"] and is_muse else None)
    chaos = Chaos(cfg["chaos"]["scripted"] if cfg["chaos"]["enabled"] else None,
                  llm=getattr(planner, "client", None) if not offline else None)
    return ReActAgent(world, perception, planner, imagination=imagination, chaos=chaos,
                      dashboard=Dashboard(cfg["viz"]["rerun"]),
                      max_steps=cfg["max_steps"], max_replans=cfg["max_replans"],
                      planner_name=cfg["planning"]["planner"] + ("+imagine" if imagination else ""),
                      world_name=cfg["world"])
