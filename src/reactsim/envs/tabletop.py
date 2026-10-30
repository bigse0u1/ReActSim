"""ReActTabletop-v1: Panda + everyday objects + a plate, in one of several scenes.

Scenes: "kitchen" (banana, mug, soup can) and "home" (orange, soup can, cup),
the second matching what is on hand for the human demonstration videos.

Real object shapes matter twice: SAM 3.1 has something non-trivial to segment,
and different shapes favour different grasps, which gives Imagination real choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda

from reactsim.agents import xarm_ability
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
PLATE = ObjectSpec("029_plate", "plate", "red", (0.0, 0.17), 0.0147)
SCENES = {
    "kitchen": {
        "banana": ObjectSpec("011_banana", "banana", "yellow", (0.05, -0.06), 0.0189, np.pi / 4),
        # With the handle turned away the mug reads as a cup: "mug" finds nothing, "cup" gets IoU 0.99.
        "mug": ObjectSpec("025_mug", "mug", "red", (-0.10, -0.14), 0.0407, np.pi, sam_prompt="cup"),
        "can": ObjectSpec("005_tomato_soup_can", "can", "red and white", (0.12, -0.20), 0.0513),
        "plate": PLATE,
    },
    # Stand-ins for the real objects in the demo videos: drink can -> soup can,
    # mandarin -> YCB orange, paper cup -> YCB cup (7.2 cm, fits the 8 cm gripper).
    "home": {
        # The YCB orange renders like a ball: "orange" and "fruit" find nothing, "ball" gets IoU 0.99.
        "orange": ObjectSpec("017_orange", "orange", "orange", (0.05, -0.06), 0.0365, sam_prompt="ball"),
        "can": ObjectSpec("005_tomato_soup_can", "can", "red and white", (0.12, -0.20), 0.0513),
        "cup": ObjectSpec("065-d_cups", "cup", "yellow", (-0.10, -0.14), 0.036),
        "plate": PLATE,
    },
}
# Cluttered: the kitchen objects, but the mug and the can stand right next to the banana in
# random directions, so some approaches to it are blocked (see _place_clutter).
SCENES["clutter"] = dict(SCENES["kitchen"])
# Neighbour centre distance from the banana (m). Measured on 8 layouts each: at 10-12 cm the
# robot-side grasps solve 3/8 and all approach directions 6/8; at 12-14 cm 4/8 vs 8/8; at
# 14-16 cm 7/8 vs 7/8 (nothing is in the way). 12-14 cm: "a few things nearby".
CLUTTER_RADIUS = (0.12, 0.14)
OBJECTS = {name: spec for scene in SCENES.values() for name, spec in scene.items()}   # name -> spec
# Chaos ball: a YCB baseball thrown at objects, even out of the gripper. It floats (no
# gravity) out of view until thrown, then flies a real ballistic arc.
CHAOS_BALL_ID = "055_baseball"
CHAOS_BALL_RADIUS = 0.037
CHAOS_BALL_PARK = (3.0, 3.0, 0.5)    # behind both cameras
PLATE_RADIUS = 0.13
PLATE_TOP = 2 * PLATE.half_height   # rim height; the plate is concave inside
ON_PLATE_LIFT = 0.003   # an object on the plate sits at least this much higher than on the table
XY_JITTER = 0.03
ROBOT_BASE_X = {"panda": -0.615, xarm_ability.UID: xarm_ability.BASE_X}


@register_env("ReActTabletop-v1", max_episode_steps=2000)
class ReActTabletopEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", xarm_ability.UID]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, scene="kitchen", **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_name = scene
        self.specs = SCENES[scene]
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

    @property
    def base_x(self) -> float:
        return ROBOT_BASE_X[self.robot_uids]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[self.base_x, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.objects = {}
        for name, spec in self.specs.items():
            builder = get_actor_builder(self.scene, id=f"ycb:{spec.ycb_id}")
            builder.initial_pose = sapien.Pose(p=[*spec.xy, spec.half_height])
            # The plate is fixed so a dropped object cannot shove it away.
            self.objects[name] = (builder.build_kinematic(name=name) if name == "plate"
                                  else builder.build(name=name))
        # Kept out of self.objects so perception and planning never see it.
        builder = get_actor_builder(self.scene, id=f"ycb:{CHAOS_BALL_ID}")
        builder.initial_pose = sapien.Pose(p=CHAOS_BALL_PARK)
        self.chaos_ball = builder.build(name="chaos_ball")
        for body in self.chaos_ball._bodies:
            body.disable_gravity = True

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self.robot_uids == xarm_ability.UID:   # the table builder only knows stock robots
                rest = self.agent.keyframes["rest"].qpos[:7]
                self.agent.reset(xarm_ability.hand_qpos(rest))
                self.agent.robot.set_pose(sapien.Pose(p=[self.base_x, 0, 0]))
            for name, spec in self.specs.items():
                if name == "plate":
                    continue
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = torch.tensor(spec.xy) + (torch.rand((b, 2)) - 0.5) * 2 * XY_JITTER
                xyz[:, 2] = spec.half_height
                yaw = (torch.rand(b) * 2 - 1) * spec.yaw_range
                q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], 1)
                self.objects[name].set_pose(Pose.create_from_pq(xyz, q))
            if self.scene_name == "clutter":
                self._place_clutter(b)

    def _place_clutter(self, b: int) -> None:
        """Mug and can around the banana, at random bearings at least 100 deg apart."""
        centre = self.objects["banana"].pose.p[:, :2]
        a1 = torch.rand(b) * 2 * np.pi
        a2 = a1 + np.deg2rad(100) + torch.rand(b) * np.deg2rad(160)
        for name, ang in (("mug", a1), ("can", a2)):
            r = CLUTTER_RADIUS[0] + torch.rand(b) * (CLUTTER_RADIUS[1] - CLUTTER_RADIUS[0])
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = centre + torch.stack([torch.cos(ang), torch.sin(ang)], 1) * r[:, None]
            xyz[:, 2] = self.specs[name].half_height
            yaw = torch.rand(b) * 2 * np.pi
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], 1)
            self.objects[name].set_pose(Pose.create_from_pq(xyz, q))

    @property
    def default_target(self) -> str:
        return next(iter(self.specs))

    def evaluate(self):
        # Default task: the scene's first object on the plate. The agent uses its own
        # Goal-based checker; this keeps ManiSkill's success flag meaningful for videos.
        name = self.default_target
        obj = self.objects[name].pose.p
        plate = self.objects["plate"].pose.p
        on_xy = torch.linalg.norm(obj[:, :2] - plate[:, :2], dim=1) < PLATE_RADIUS * 0.8
        above = obj[:, 2] > self.specs[name].half_height + ON_PLATE_LIFT
        grasped = self.agent.is_grasping(self.objects[name])
        return {"success": on_xy & above & ~grasped}

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose)

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
