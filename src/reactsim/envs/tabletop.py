"""ReActTabletop-v1: Panda + three colored cubes + a plate.

Distractor cubes matter: with a single cube, language grounding is trivial and
SAM 3.1 contributes nothing a judge can see.

Written against the ManiSkill 3 API (PickCube-v1 is the reference). Verify on
first install; the TODOs mark the spots most likely to need adjustment.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

CUBE_HALF = 0.02
PLATE_HALF = 0.06
PLATE_THICK = 0.005

CUBES = {
    "red_cube": ([0.9, 0.1, 0.1, 1.0], [0.10, 0.15]),
    "blue_cube": ([0.1, 0.2, 0.9, 1.0], [0.15, -0.05]),
    "green_cube": ([0.1, 0.8, 0.2, 1.0], [0.00, -0.20]),
}
PLATE_XY = [0.25, 0.20]


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
        for name, (color, xy) in CUBES.items():
            self.objects[name] = actors.build_cube(
                self.scene, half_size=CUBE_HALF, color=color, name=name,
                initial_pose=sapien.Pose(p=[*xy, CUBE_HALF]),
            )
        self.objects["plate"] = actors.build_box(
            self.scene, half_sizes=[PLATE_HALF, PLATE_HALF, PLATE_THICK],
            color=[0.95, 0.95, 0.95, 1.0], name="plate", body_type="kinematic",
            initial_pose=sapien.Pose(p=[*PLATE_XY, PLATE_THICK]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            for name, (_, xy) in CUBES.items():
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.tensor(xy) + (torch.rand((b, 2)) - 0.5) * 0.06
                xyz[:, 2] = CUBE_HALF
                self.objects[name].set_pose(Pose.create_from_pq(xyz, [1, 0, 0, 0]))

    def evaluate(self):
        # Default task: red cube on plate. The agent uses its own Goal-based checker;
        # this keeps ManiSkill's success flag meaningful for recorded videos.
        cube = self.objects["red_cube"].pose.p
        plate = self.objects["plate"].pose.p
        on_xy = torch.all(torch.abs(cube[:, :2] - plate[:, :2]) < PLATE_HALF, dim=1)
        above = cube[:, 2] > plate[:, 2] + PLATE_THICK
        grasped = self.agent.is_grasping(self.objects["red_cube"])
        return {"success": on_xy & above & ~grasped}

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
