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
GRASP_PITCH_DEG = {"angled": 50.0, "side": 0.0}   # "top" is straight down
SIDE_GRASP_DZ = 0.01   # grip slightly above centre so the hand clears the table
APPROACH_OFFSET = 0.08
CLEAR_VIEW_OFFSET = (0.40, 0.0, 0.35)   # from the robot base: behind the objects as the camera sees them
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

    def release(self) -> None:
        self.planner.open_gripper()

    def idle(self, steps: int) -> None:
        """Keep the current joint positions (and gripper) for `steps` control steps."""
        for _ in range(steps):
            self.raw_env.step(self.env.agent.robot.get_qpos()[0, :8].cpu().numpy())

    def after_replay(self, commands) -> None:
        """Commands were sent without the planner: sync its gripper state with the last one."""
        self.planner.gripper_state = float(commands[-1][-1])

    def reset_planner(self):
        """Call after env.reset / set_state_dict so the planner sees the new robot state."""
        self.planner.close()
        self.planner = self._make_planner()

    @staticmethod
    def _ok(res) -> bool:
        return not (isinstance(res, int) and res == -1)

    def _approach(self, p: np.ndarray, grasp: str, approach_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Approach direction and closing hint. Non-top grasps come from the robot's side,
        pitched down; measured on ReActTabletop: 50° lifts reliably, 0° often slips."""
        if grasp == "top":
            closing = self.env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
            return np.array([0.0, 0.0, -1.0]), closing
        base = self.env.agent.robot.pose.sp.p
        h = np.array([p[0] - base[0], p[1] - base[1], 0.0])
        h /= np.linalg.norm(h)
        if approach_deg:   # come in from another side of the object
            c, s = np.cos(np.deg2rad(approach_deg)), np.sin(np.deg2rad(approach_deg))
            h = np.array([c * h[0] - s * h[1], s * h[0] + c * h[1], 0.0])
        pitch = np.deg2rad(GRASP_PITCH_DEG[grasp])
        approaching = h * np.cos(pitch) + np.array([0.0, 0.0, -np.sin(pitch)])
        closing = np.cross(approaching, [0.0, 0.0, 1.0])
        return approaching, closing / np.linalg.norm(closing)

    def pick(self, actor, grasp: str = "top", box=None, approach_deg: float | None = None) -> SkillResult:
        """box: the object's box as perceived (estimate_box); None uses the simulator's geometry."""
        t0 = time.time()
        env = self.env
        obb = box if box is not None else get_actor_obb(actor)
        p = np.array(obb.primitive.transform)[:3, 3] if box is not None else actor.pose.sp.p
        approaching, target_closing = self._approach(p, grasp, approach_deg)
        info = compute_grasp_info_by_obb(obb, approaching=approaching,
                                         target_closing=target_closing, depth=FINGER_LENGTH)
        if grasp == "side":
            # The box picks its shortest remaining axis, which on a low object (banana, mug) is
            # the height: the fingers would close top-to-bottom and the hand stand on its edge
            # against the table. Close them horizontally, across the object, instead.
            info["closing"] = target_closing
        center = p + np.array([0.0, 0.0, 0.0 if grasp == "top" else SIDE_GRASP_DZ])
        grasp_pose = env.agent.build_grasp_pose(approaching, info["closing"], center)
        reach_pose = grasp_pose * sapien.Pose([0, 0, -APPROACH_OFFSET])

        self.planner.open_gripper()
        # A straight-line (screw) move cannot make large wrist rotations; fall back to RRT.
        if not (self._ok(self.planner.move_to_pose_with_screw(reach_pose))
                or self._ok(self.planner.move_to_pose_with_RRTConnect(reach_pose))):
            return SkillResult(False, "no path to pre-grasp", time.time() - t0)
        # RRT reaches the pre-grasp in a random arm configuration, from which the straight
        # final move can be infeasible; then plan it with RRT too (short, so nearly straight).
        if not (self._ok(self.planner.move_to_pose_with_screw(grasp_pose))
                or self._ok(self.planner.move_to_pose_with_RRTConnect(grasp_pose))):
            return SkillResult(False, "no path to grasp", time.time() - t0)
        self.planner.close_gripper()
        lift = sapien.Pose([0, 0, LIFT_HEIGHT]) * grasp_pose
        if not self._ok(self.planner.move_to_pose_with_screw(lift)):
            return SkillResult(False, "lift failed", time.time() - t0)
        # Whether the object actually came along is checked by the diagnoser, not here.
        return SkillResult(True, "pick motion finished", time.time() - t0)

    def place_on(self, ref_xyz: np.ndarray, drop_height: float = 0.05, q: np.ndarray | None = None) -> SkillResult:
        """q: the gripper orientation to put the object down in (default: as it is held)."""
        t0 = time.time()
        tcp_q = self.env.agent.tcp.pose.sp.q if q is None else q
        above = sapien.Pose(p=ref_xyz + np.array([0, 0, LIFT_HEIGHT]), q=tcp_q)
        down = sapien.Pose(p=ref_xyz + np.array([0, 0, drop_height]), q=tcp_q)
        for pose, msg in ((above, "transport"), (down, "descend")):
            # A side-held object keeps the wrist horizontal, where straight moves often hit joint limits.
            if not (self._ok(self.planner.move_to_pose_with_screw(pose))
                    or self._ok(self.planner.move_to_pose_with_RRTConnect(pose))):
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

    def clear_view(self) -> bool:
        """Move the arm back toward its base, out of the camera's line of sight to the table.
        Right after a slip the hand hovers over the object and hides most of it."""
        tcp = self.env.agent.tcp.pose.sp
        base = self.env.agent.robot.pose.sp.p
        target = base + np.array(CLEAR_VIEW_OFFSET)
        if np.linalg.norm(tcp.p - target) < 0.05:
            return True
        pose = sapien.Pose(p=target, q=tcp.q)
        return (self._ok(self.planner.move_to_pose_with_screw(pose))
                or self._ok(self.planner.move_to_pose_with_RRTConnect(pose)))

    def home(self) -> SkillResult:
        t0 = time.time()
        p = self.env.agent.tcp.pose.sp
        self.planner.move_to_pose_with_screw(sapien.Pose(p=[p.p[0], p.p[1], 0.3], q=p.q))
        return SkillResult(True, "home", time.time() - t0)
