"""Primitive skills on top of ManiSkill's mplib motion planner.

Muse Spark never touches joints: it picks a skill + arguments, these functions
turn that into collision-aware trajectories. Adapted from ManiSkill's
motionplanning/panda/solutions/pick_cube.py (Apache-2.0) — credit in README.
"""

from __future__ import annotations

import time

import numpy as np
import sapien

from reactsim.types import SkillResult

try:  # module path moved between ManiSkill releases
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
except ImportError:  # pragma: no cover
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (  # TODO: verify
        compute_grasp_info_by_obb,
        get_actor_obb,
    )
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

FINGER_LENGTH = 0.025
APPROACH_OFFSET = 0.08
LIFT_HEIGHT = 0.15


class PandaSkills:
    def __init__(self, env, vis: bool = False):
        self.env = env.unwrapped
        self.raw_env = env
        self.vis = vis
        self.planner = self._make_planner()

    def _make_planner(self):
        return PandaArmMotionPlanningSolver(
            self.raw_env, debug=False, vis=self.vis,
            base_pose=self.env.agent.robot.pose,
            visualize_target_grasp_pose=self.vis, print_env_info=False,
        )

    def reset_planner(self):
        """Call after env.reset / set_state_dict so the planner sees the new robot state."""
        self.planner.close()
        self.planner = self._make_planner()

    @staticmethod
    def _ok(res) -> bool:
        return not (isinstance(res, int) and res == -1)

    def pick(self, actor, grasp: str = "top") -> SkillResult:
        t0 = time.time()
        env = self.env
        obb = get_actor_obb(actor)
        if grasp == "side":
            approaching = np.array([1.0, 0.0, 0.0])   # TODO: choose side from robot->object direction
        else:
            approaching = np.array([0.0, 0.0, -1.0])
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        info = compute_grasp_info_by_obb(obb, approaching=approaching,
                                         target_closing=target_closing, depth=FINGER_LENGTH)
        grasp_pose = env.agent.build_grasp_pose(approaching, info["closing"], actor.pose.sp.p)
        reach_pose = grasp_pose * sapien.Pose([0, 0, -APPROACH_OFFSET])

        self.planner.open_gripper()
        if not self._ok(self.planner.move_to_pose_with_screw(reach_pose)):
            return SkillResult(False, "no path to pre-grasp", time.time() - t0)
        if not self._ok(self.planner.move_to_pose_with_screw(grasp_pose)):
            return SkillResult(False, "no path to grasp", time.time() - t0)
        self.planner.close_gripper()
        lift = sapien.Pose([0, 0, LIFT_HEIGHT]) * grasp_pose
        if not self._ok(self.planner.move_to_pose_with_screw(lift)):
            return SkillResult(False, "lift failed", time.time() - t0)
        # Whether the object actually came along is checked by the diagnoser, not here.
        return SkillResult(True, "pick motion finished", time.time() - t0)

    def place_on(self, ref_xyz: np.ndarray, drop_height: float = 0.05) -> SkillResult:
        t0 = time.time()
        tcp_q = self.env.agent.tcp.pose.sp.q
        above = sapien.Pose(p=ref_xyz + np.array([0, 0, LIFT_HEIGHT]), q=tcp_q)
        down = sapien.Pose(p=ref_xyz + np.array([0, 0, drop_height]), q=tcp_q)
        for pose, msg in ((above, "transport"), (down, "descend")):
            if not self._ok(self.planner.move_to_pose_with_screw(pose)):
                return SkillResult(False, f"{msg} failed", time.time() - t0)
        self.planner.open_gripper()
        self.planner.move_to_pose_with_screw(above)
        return SkillResult(True, "place motion finished", time.time() - t0)

    def push_aside(self, actor, dy: float = 0.10) -> SkillResult:
        t0 = time.time()
        p = actor.pose.sp.p
        tcp_q = self.env.agent.tcp.pose.sp.q
        self.planner.close_gripper()
        start = sapien.Pose(p=p + np.array([0, -0.06, 0.02]), q=tcp_q)
        end = sapien.Pose(p=p + np.array([0, dy, 0.02]), q=tcp_q)
        for pose in (sapien.Pose(p=start.p + [0, 0, 0.1], q=tcp_q), start, end):
            if not self._ok(self.planner.move_to_pose_with_screw(pose)):
                return SkillResult(False, "push path failed", time.time() - t0)
        return SkillResult(True, "push finished", time.time() - t0)

    def home(self) -> SkillResult:
        t0 = time.time()
        p = self.env.agent.tcp.pose.sp
        self.planner.move_to_pose_with_screw(sapien.Pose(p=[p.p[0], p.p[1], 0.3], q=p.q))
        return SkillResult(True, "home", time.time() - t0)
