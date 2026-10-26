"""World implementation backed by ManiSkill 3 + the ReActTabletop-v1 env."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import reactsim.envs.tabletop  # noqa: F401  (registers ReActTabletop-v1)
from reactsim.control.skills import PandaSkills
from reactsim.envs.tabletop import CUBE_HALF, PLATE_HALF, PLATE_THICK
from reactsim.types import (
    Action,
    ActionType,
    Goal,
    Observation,
    ObjectState,
    PerturbEvent,
    SkillResult,
)

COLORS = {"red_cube": "red", "blue_cube": "blue", "green_cube": "green", "plate": "white"}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class ManiSkillWorld:
    def __init__(self, env_id="ReActTabletop-v1", render_mode="rgb_array",
                 sim_backend="cpu", record_dir: str | None = None, vis=False):
        env = gym.make(
            env_id,
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",   # required by the mplib motion planner
            render_mode=render_mode,
            sim_backend=sim_backend,
            num_envs=1,
        )
        if record_dir:
            from mani_skill.utils.wrappers.record import RecordEpisode
            env = RecordEpisode(env, output_dir=record_dir, save_trajectory=False,
                                save_video=True, video_fps=30)
        self.env = env
        self._record = bool(record_dir)
        self.vis = vis
        self.skills: PandaSkills | None = None
        self._last_obs = None

    @property
    def u(self):
        return self.env.unwrapped

    def reset(self, seed: int) -> None:
        self._last_obs, _ = self.env.reset(seed=seed)
        if self.skills is None:
            self.skills = PandaSkills(self.env, vis=self.vis)
        else:
            self.skills.reset_planner()

    def _holding(self) -> str | None:
        for name, actor in self.u.objects.items():
            if name != "plate" and bool(self.u.agent.is_grasping(actor)[0]):
                return name
        return None

    def observe(self) -> Observation:
        obs = self.u.get_obs()
        cam = obs["sensor_data"]["base_camera"]
        param = obs["sensor_param"]["base_camera"]
        holding = self._holding()
        gt = {}
        for name, actor in self.u.objects.items():
            gt[name] = ObjectState(name, _np(actor.pose.p[0]), COLORS[name],
                                   "plate" if name == "plate" else "cube", held=name == holding)
        return Observation(
            rgb=_np(cam["rgb"][0]).astype(np.uint8),
            depth=_np(cam["depth"][0, ..., 0]).astype(np.float32) / 1000.0,  # mm -> m
            intrinsic=_np(param["intrinsic_cv"][0]),
            extrinsic=_np(param["extrinsic_cv"][0]),
            segmentation=_np(cam["segmentation"][0, ..., 0]),
            gt_objects=gt,
            gripper_holding=holding,
        )

    def segmentation_ids(self) -> dict[str, int]:
        """Actor name -> segmentation id, for the oracle segmenter."""
        return {name: int(a.per_scene_id[0]) for name, a in self.u.objects.items()}

    def execute(self, action: Action) -> SkillResult:
        s, objs = self.skills, self.u.objects
        t = action.type
        if t == ActionType.PICK:
            return s.pick(objs[action.target], action.grasp)
        if t == ActionType.PLACE_ON:
            ref = _np(objs[action.reference].pose.p[0])
            return s.place_on(ref + np.array([0, 0, PLATE_THICK + CUBE_HALF]))
        if t == ActionType.PUSH_ASIDE:
            return s.push_aside(objs[action.target])
        if t == ActionType.HOME:
            return s.home()
        return SkillResult(True, t.value)

    def perturb(self, event: PerturbEvent) -> None:
        import sapien
        if event.kind == "drop":
            s = self.skills
            s.planner.open_gripper()
            return
        actor = self.u.objects.get(event.object)
        if actor is None:
            return
        if self._holding() == event.object:
            self.skills.planner.open_gripper()   # a knock frees a held object
        p = _np(actor.pose.p[0])
        actor.set_pose(sapien.Pose(p=[p[0] + event.dx, p[1] + event.dy, CUBE_HALF]))
        if event.kind == "knock":
            # TODO: replace the teleport with an impulse for a more physical look
            actor.set_linear_velocity(np.array([[event.dx * 2, event.dy * 2, 0.0]]))
        for _ in range(20):   # let it settle
            self.env.step(_np(self.u.agent.robot.get_qpos()[:, :8]))

    def is_goal_satisfied(self, goal: Goal) -> bool:
        objs = self.u.objects
        obj, ref = objs.get(goal.target), objs.get(goal.reference)
        if obj is None or ref is None or self._holding() == goal.target:
            return False
        po, pr = _np(obj.pose.p[0]), _np(ref.pose.p[0])
        resting = np.linalg.norm(_np(obj.linear_velocity[0])) < 0.05
        return bool(np.all(np.abs(po[:2] - pr[:2]) < PLATE_HALF) and po[2] > pr[2] and resting)

    def snapshot(self):
        return self.u.get_state_dict()

    def restore(self, snap) -> None:
        self.u.set_state_dict(snap)
        self.skills.reset_planner()

    def set_recording(self, on: bool) -> None:
        """Pause video capture while imagining so the video shows only real execution."""
        if hasattr(self.env, "_save_video"):
            self.env._save_video = on and self._record

    def close(self) -> None:
        if hasattr(self.env, "flush_video"):
            self.env.flush_video()
        self.env.close()

    def frame(self):
        f = self.env.render()
        return _np(f[0]) if f is not None and getattr(f, "ndim", 0) == 4 else f
