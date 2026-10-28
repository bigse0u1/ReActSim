"""Skills for the xArm7 + Ability hand, with the same interface as PandaSkills.

Grasps (the agent's vocabulary, so Muse Spark and Imagination use them unchanged):
  top    five fingers open around the object from above, close until each pad touches,
         then squeeze. Measured (5 seeds, held 3 s): orange 4/5, can 5/5, cup 3/5.
  side   palm against the object's side, fingers wrap around it.
  angled the same five-finger enclose, but the hand comes in pitched 50 deg down from the
         robot's side: an alternative when the path straight down is blocked.

Every other object is a point-cloud obstacle for the planner, and so is the target
until the hand closes in on it (otherwise the forearm knocks it over on the way).
"""

from __future__ import annotations

import time

import mplib
import numpy as np
import sapien
from transforms3d.quaternions import mat2quat

from reactsim.agents.xarm_ability import BASE_X, MIMIC_B, MIMIC_K, build_assets
from reactsim.types import SkillResult

try:  # module path moved between ManiSkill releases
    from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
except ImportError:  # pragma: no cover
    from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb

# Hand shapes: [thumb opposition, index, middle, ring, pinky, thumb flex] (q1 targets)
OPEN = {"top": np.array([1.8, 0.0, 0.0, 0.0, 0.0, 0.0]), "side": np.zeros(6)}
CLOSED = {"top": np.array([1.8, 1.0, 1.0, 1.0, 1.0, 1.0]), "side": np.array([1.2, 1.6, 1.6, 1.6, 1.6, 0.0])}
OPEN["angled"], CLOSED["angled"] = OPEN["top"], CLOSED["top"]   # same hand shape, tilted approach
ANGLED_PITCH_DEG = 50.0
RELAXED = np.zeros(6)
# Measured fingertip geometry in the wrist frame with OPEN["top"]: the thumb tip and the
# midpoint of the index/middle tips, 16 cm apart. The hand is turned so this line is level.
TOP_THUMB = np.array([-0.011, 0.096, 0.061])
TOP_FINGERS = np.array([-0.019, -0.013, 0.175])
SIDE_CENTER = np.array([-0.005, 0.045, 0.10])   # where a can sits in the palm for "side"
SIDE_ABOVE_CENTER = 0.025       # side grasp: keep the hand's lower edge off the table
STANDOFF = {"top": 0.12, "angled": 0.12, "side": 0.08}
TRANSIT_RISE = 0.15
LIFT_HEIGHT = 0.15
FINGER_RATE = 0.03              # rad per control step
CONTACT_N = 0.5                 # N on a finger pad counts as touching
SQUEEZE = 0.5                   # rad past first contact: the PD then presses at its force limit
OBSTACLE_PAD = 0.02
SETTLE_STEPS = 15
RRT_ATTEMPTS = 3
HEADINGS_DEG = (0, 45, -45, 90, -90, 135, -135, 180)
ANGLED_HEADINGS_DEG = (0, 30, -30)   # tilted from the robot's side, so only near-frontal headings
PADS = {0: "thumb_L2", 1: "index_L2", 2: "middle_L2", 3: "ring_L2", 4: "pinky_L2"}


class HandSkills:
    def __init__(self, env, vis: bool = False):
        self.env = env.unwrapped
        self.raw_env = env          # steps go through the caller's wrappers (recording, action log)
        self.robot = self.env.agent.robot
        self.links = {l.name: l for l in self.robot.get_links()}
        self.wrist = self.links["base"]
        self.hand = RELAXED.copy()
        self.grasp = None
        self.arm_target = None
        self.planner = self._make_planner()

    def _make_planner(self):
        paths = build_assets()
        pl = mplib.Planner(urdf=str(paths["plan"]), srdf=str(paths["srdf"]), move_group="base")
        pl.set_base_pose(np.r_[[BASE_X, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        return pl

    def reset_planner(self):
        """Call after env.reset / set_state_dict: re-read the robot state."""
        q, t = self._q(), self._finger_target()
        self.arm_target = q[:7]
        self.hand = np.r_[-t[0], t[1:5], t[5]]

    def release(self) -> None:
        self.hand = RELAXED.copy()
        self._hold(10)

    def idle(self, steps: int) -> None:
        self._hold(steps)

    def after_replay(self, commands) -> None:
        self.reset_planner()

    # ---- low level -------------------------------------------------------------------
    def _q(self) -> np.ndarray:
        return self.robot.get_qpos()[0].cpu().numpy()

    def _finger_target(self) -> np.ndarray:
        return self.env.agent.controller.controllers["gripper"]._target_qpos[0].cpu().numpy()

    def _plan_q(self) -> np.ndarray:
        """Joint vector in the planner's order (it lists finger joints differently)."""
        q, names = self._q(), [j.name for j in self.robot.get_active_joints()]
        return np.array([q[names.index(n)] for n in self.planner.user_joint_names])

    def _step(self, arm: np.ndarray) -> None:
        """One control step: arm to `arm`, each finger a bounded step toward self.hand
        (q2 follows q1 as the removed mimic would)."""
        # The hand controller takes normalised deltas (+-0.1 rad) on its own target. Steer that
        # target, not the measured angle: a finger blocked by the object lags its target, and
        # deltas from the measured angle would wind the target up past the fingers.
        h, r = self._finger_target(), FINGER_RATE
        t1 = np.clip(-self.hand[0] - h[0], -r, r) / 0.1
        f1 = np.clip(self.hand[1:5] - h[1:5], -r, r) / 0.1
        t2 = np.clip(self.hand[5] - h[5], -r, r) / 0.1
        f2 = np.clip(MIMIC_K * h[1:5] + MIMIC_B - h[6:10], -0.08, 0.08) / 0.1
        self.raw_env.step(np.concatenate([arm, [t1], f1, [t2], f2]))

    def _hold(self, n: int) -> None:
        # Hold the last commanded arm pose: re-targeting the measured one lets it creep.
        arm = self._q()[:7] if self.arm_target is None else self.arm_target
        for _ in range(n):
            self._step(arm)

    def _plan(self, pose: sapien.Pose, start: np.ndarray, obstacles: bool = True):
        """Plan to `pose` from `start` (planner joint order); None if there is no path."""
        goal = np.r_[pose.p, pose.q]
        kw = dict(time_step=self.env.control_timestep, use_point_cloud=obstacles)
        res = self.planner.plan_screw(goal, start, **kw)
        # RRT is random and gives up after its time budget, so one miss is not "no path".
        for _ in range(RRT_ATTEMPTS):
            if res["status"] == "Success":
                break
            res = self.planner.plan_qpos_to_pose(goal, start, **kw)
        return res if res["status"] == "Success" else None

    def _follow(self, res) -> None:
        for p in res["position"]:
            self._step(p[:7])
        self.arm_target = res["position"][-1][:7]

    def _move(self, pose: sapien.Pose, obstacles: bool = True) -> bool:
        res = self._plan(pose, self._plan_q(), obstacles)
        if res is not None:
            self._follow(res)
        return res is not None

    def _set_obstacles(self, skip: set[str]) -> None:
        pts = []
        for name, a in self.env.objects.items():
            if name in skip:
                continue
            obb = get_actor_obb(a)
            T = np.array(obb.primitive.transform)
            ext = np.array(obb.primitive.extents) / 2 + OBSTACLE_PAD
            g = np.stack(np.meshgrid(*[np.linspace(-e, e, 6) for e in ext]), -1).reshape(-1, 3)
            pts.append((T[:3, :3] @ g.T).T + T[:3, 3])
        if pts:
            self.planner.update_point_cloud(np.concatenate(pts), radius=0.01)   # world frame

    def _name_of(self, actor) -> str:
        return next(n for n, a in self.env.objects.items() if a is actor)

    # ---- grasp geometry --------------------------------------------------------------
    def _wrist_pose(self, center, heading: np.ndarray, grasp: str, standoff: float = 0.0) -> sapien.Pose:
        if grasp in ("top", "angled"):
            u = TOP_FINGERS - TOP_THUMB
            u /= np.linalg.norm(u)
            a = np.array([0, 0, 1.0]) - u[2] * u
            a /= np.linalg.norm(a)
            local = np.stack([u, a, np.cross(u, a)], 1)
            uw = np.cross([0, 0, 1.0], heading)        # thumb-to-fingers level, across the approach
            aw = np.array([0, 0, -1.0])                # fingers point down
            if grasp == "angled":                      # ... or down and away from the robot
                pitch = np.deg2rad(ANGLED_PITCH_DEG)
                aw = heading * np.cos(pitch) + np.array([0, 0, -np.sin(pitch)])
            R = np.stack([uw, aw, np.cross(uw, aw)], 1) @ local.T
            mid = (TOP_THUMB + TOP_FINGERS) / 2
            return sapien.Pose(p=center - R @ mid - aw * standoff, q=mat2quat(R))
        # side: palm faces the object, thumb up, fingers wrap horizontally
        x, y = np.array([0, 0, -1.0]), heading
        R = np.stack([x, y, np.cross(x, y)], 1)
        return sapien.Pose(p=center - R @ SIDE_CENTER - y * standoff, q=mat2quat(R))

    def _close_on_contact(self, actor, grasp: str, max_steps: int = 80) -> list[str]:
        """Close each finger a small step at a time; a finger that touches stops there and
        braces the object instead of pushing it away. Then squeeze."""
        fingers = {i: PADS[i] for i in range(5)}
        hand_index = {0: 5, 1: 1, 2: 2, 3: 3, 4: 4}     # the thumb closes by flexing (index 5)
        touching = set()
        for _ in range(max_steps):
            for i, pad in fingers.items():
                if i in touching:
                    continue
                f = self.env.scene.get_pairwise_contact_forces(self.links[pad], actor)
                k = hand_index[i]
                if float(np.linalg.norm(f.cpu().numpy())) > CONTACT_N:
                    touching.add(i)
                    self.hand[k] = self._q()[7:][k] * (-1 if k == 0 else 1)
                else:
                    self.hand[k] = min(self.hand[k] + FINGER_RATE, CLOSED[grasp][k])
            if len(touching) == len(fingers):
                break
            self._hold(1)
        for i in fingers:
            self.hand[hand_index[i]] += SQUEEZE
        self._hold(15)
        return [PADS[i].split("_")[0] for i in sorted(touching)]

    # ---- skills ----------------------------------------------------------------------
    def pick(self, actor, grasp: str = "top", box=None) -> SkillResult:
        """box: the object's box as perceived (estimate_box); None uses the simulator's pose."""
        t0 = time.time()
        if grasp not in OPEN:
            return SkillResult(False, f"this hand has no {grasp} grasp", time.time() - t0)
        name = self._name_of(actor)
        p0 = np.array(box.primitive.transform)[:3, 3] if box is not None else actor.pose.sp.p.copy()
        center = p0 + [0, 0, SIDE_ABOVE_CENTER if grasp == "side" else 0.0]
        base = np.array([BASE_X, 0.0])
        h = np.r_[p0[:2] - base, 0.0]
        h /= np.linalg.norm(h)

        self.hand = OPEN[grasp].copy()
        self._hold(25)
        # A top grasp works from any heading: turn the hand until it fits between the
        # other objects (the thumb sticks out 11 cm). The side grasp faces the object.
        headings = {"top": HEADINGS_DEG, "angled": ANGLED_HEADINGS_DEG}.get(grasp, (0,))
        # Plan all three legs for a heading before moving, so a heading that fails part-way
        # leaves no motion behind (and the executed commands can be replayed exactly).
        for deg in headings:
            c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
            heading = np.array([c * h[0] - s * h[1], s * h[0] + c * h[1], 0.0])
            grasp_pose = self._wrist_pose(center, heading, grasp)
            pre = self._wrist_pose(center, heading, grasp, STANDOFF[grasp])
            above = sapien.Pose(p=pre.p + [0, 0, TRANSIT_RISE], q=pre.q)
            legs, start = [], self._plan_q()
            for pose, skip in ((above, {"plate"}), (pre, {"plate"}), (grasp_pose, {"plate", name})):
                self._set_obstacles(skip)
                res = self._plan(pose, start)
                if res is None:
                    break
                legs.append(res)
                start = start.copy()
                start[:7] = res["position"][-1][:7]
            if len(legs) == 3:
                break
        else:
            return SkillResult(False, "no path to grasp", time.time() - t0)
        for res in legs:
            self._follow(res)

        self.grasp = grasp
        self._close_on_contact(actor, grasp)
        self._set_obstacles(skip={"plate", name})
        if not self._move(sapien.Pose(p=grasp_pose.p + [0, 0, LIFT_HEIGHT], q=grasp_pose.q)):
            return SkillResult(False, "lift failed", time.time() - t0)
        # Whether the object came along is checked by the diagnoser, not here.
        return SkillResult(True, "pick motion finished", time.time() - t0)

    def place_on(self, ref_xyz: np.ndarray, drop_height: float = 0.0) -> SkillResult:
        """Carry the held object over ref_xyz, lower the wrist to ref_xyz (+drop_height), open."""
        t0 = time.time()
        q = self.wrist.pose.sp.q
        held = [n for n, a in self.env.objects.items() if bool(self.env.agent.is_grasping(a)[0])]
        self._set_obstacles(skip={"plate", *held})
        above = sapien.Pose(p=ref_xyz + [0, 0, TRANSIT_RISE], q=q)
        down = sapien.Pose(p=ref_xyz + [0, 0, drop_height], q=q)
        for pose, msg in ((above, "transport"), (down, "descend")):
            if not self._move(pose):
                return SkillResult(False, f"{msg} failed", time.time() - t0)
        self.hand = OPEN.get(self.grasp, RELAXED).copy()   # open in the same shape it closed from
        self._hold(30)
        self._set_obstacles(skip={"plate"})
        self._move(above, obstacles=False)
        self._hold(SETTLE_STEPS)   # let it come to rest before anyone checks where it landed
        return SkillResult(True, "place motion finished", time.time() - t0)

    def push_aside(self, actor, dy: float = 0.10) -> SkillResult:
        return SkillResult(False, "push_aside is not implemented for the hand")

    def clear_view(self) -> bool:
        """Move the hand back toward the base, out of the camera's line of sight (see PandaSkills)."""
        target = np.array([BASE_X + 0.30, 0.0, 0.40])
        if np.linalg.norm(self.wrist.pose.sp.p - target) < 0.05:
            return True
        self._set_obstacles(skip={"plate"})
        return self._move(sapien.Pose(p=target, q=self.wrist.pose.sp.q))

    def home(self) -> SkillResult:
        t0 = time.time()
        p = self.wrist.pose.sp
        self._move(sapien.Pose(p=[p.p[0], p.p[1], 0.3], q=p.q))
        return SkillResult(True, "home", time.time() - t0)
