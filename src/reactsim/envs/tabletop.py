"""ReActTabletop-v1: Panda + everyday YCB objects (banana, mug, soup can) + a plate.

Real object shapes matter twice: SAM 3.1 has something non-trivial to segment,
and different shapes favour different grasps, which gives Imagination real choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors.common import get_actor_builder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@dataclass(frozen=True)
class ObjectSpec:
    ycb_id: str
    kind: str
    color: str
    xy: tuple[float, float]
    half_height: float        # from the YCB bbox; the model origin is the bbox centre
    yaw_range: float = 0.0    # random yaw +/- this (rad)
    sam_prompt: str = ""      # noun phrase SAM 3.1 finds reliably; defaults to the name


# Robot base is at x=-0.615; Panda reaches ~0.85 m, so keep everything within ~0.75 m of it.
OBJECTS = {
    "banana": ObjectSpec("011_banana", "banana", "yellow", (0.05, -0.06), 0.0189, np.pi / 4),
    # With the handle turned away the mug reads as a cup: "mug" finds nothing, "cup" gets IoU 0.99.
    "mug": ObjectSpec("025_mug", "mug", "red", (-0.10, -0.14), 0.0407, np.pi, sam_prompt="cup"),
    "can": ObjectSpec("005_tomato_soup_can", "can", "red and white", (0.12, -0.20), 0.0513),
    "plate": ObjectSpec("029_plate", "plate", "red", (0.0, 0.17), 0.0147),
}
PLATE_RADIUS = 0.13
PLATE_TOP = 2 * OBJECTS["plate"].half_height
XY_JITTER = 0.03


@register_env("ReActTabletop-v1", max_episode_steps=2000)
class ReActTabletopEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # Higher than PickCube's 128px so SAM masks are usable.
        pose = sapien_utils.look_at(eye=[0.55, 0.0, 0.55], target=[0.1, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 512, 512, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.7, 0.6, 0.6], target=[0.05, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 1024, 1024, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.objects = {}
        for name, spec in OBJECTS.items():
            builder = get_actor_builder(self.scene, id=f"ycb:{spec.ycb_id}")
            builder.initial_pose = sapien.Pose(p=[*spec.xy, spec.half_height])
            # The plate is fixed so a dropped object cannot shove it away.
            self.objects[name] = (builder.build_kinematic(name=name) if name == "plate"
                                  else builder.build(name=name))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            for name, spec in OBJECTS.items():
                if name == "plate":
                    continue
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.tensor(spec.xy) + (torch.rand((b, 2)) - 0.5) * 2 * XY_JITTER
                xyz[:, 2] = spec.half_height
                yaw = (torch.rand(b) * 2 - 1) * spec.yaw_range
                q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], 1)
                self.objects[name].set_pose(Pose.create_from_pq(xyz, q))

    def evaluate(self):
        # Default task: banana on plate. The agent uses its own Goal-based checker;
        # this keeps ManiSkill's success flag meaningful for recorded videos.
        obj = self.objects["banana"].pose.p
        plate = self.objects["plate"].pose.p
        on_xy = torch.linalg.norm(obj[:, :2] - plate[:, :2], dim=1) < PLATE_RADIUS * 0.8
        above = obj[:, 2] > PLATE_TOP
        grasped = self.agent.is_grasping(self.objects["banana"])
        return {"success": on_xy & above & ~grasped}

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
