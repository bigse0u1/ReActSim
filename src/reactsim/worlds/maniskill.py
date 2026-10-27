"""World implementation backed by ManiSkill 3 + the ReActTabletop-v1 env."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import reactsim.envs.tabletop  # noqa: F401  (registers ReActTabletop-v1)
from reactsim.control.skills import PandaSkills
from reactsim.envs.tabletop import (
    CHAOS_BALL_PARK, CHAOS_BALL_RADIUS, ON_PLATE_LIFT, PLATE_RADIUS, PLATE_TOP,
)
from reactsim.types import (
    Action,
    ActionType,
    Goal,
    Observation,
    ObjectState,
    PerturbEvent,
    SkillResult,
)

PLACE_CLEARANCE = 0.02   # release the held object this far above the plate
THROW_DISTANCE = 0.6     # the ball is launched this far from the target (m)
THROW_RISE = 0.2         # ... and this much higher
THROW_TIME = 0.10        # flight time (s) -> ~6 m/s; frees a held banana in 4/4 seeds. At ~4 m/s the
                         # higher arc sometimes flings the banana off the table and the mug with it.
KNOCK_END_OFFSET = 0.07  # aim this far from an elongated object's centre, to twist it
UPRIGHT_KINDS = {"can", "cup", "mug"}   # must end standing; a tipped-over cup is not "on the plate"
MAX_TILT_DEG = 30.0


class _ActionLog(gym.Wrapper):
    """Records the joint commands the motion planner sends, so an imagined skill can be
    replayed exactly instead of re-planned (RRT is random and could find another path)."""

    def __init__(self, env):
        super().__init__(env)
        self.log: list | None = None

    def step(self, action):
        if self.log is not None:
            self.log.append(np.array(action, copy=True))
        return self.env.step(action)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class ManiSkillWorld:
    def __init__(self, env_id="ReActTabletop-v1", render_mode="rgb_array",
                 sim_backend="cpu", record_dir: str | None = None, vis=False, scene="kitchen"):
        env = gym.make(
            env_id,
            scene=scene,
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
        self._actions = _ActionLog(env)   # the planner steps through this
        self.last_trace = None            # (joint commands, result) of the last execute()

    @property
    def u(self):
        return self.env.unwrapped

    def reset(self, seed: int) -> None:
        self._last_obs, _ = self.env.reset(seed=seed)
        if self.skills is None:
            self.skills = PandaSkills(self._actions, vis=self.vis)
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
            spec = self.u.specs[name]
            gt[name] = ObjectState(name, _np(actor.pose.p[0]), spec.color, spec.kind,
                                   held=name == holding, prompt=spec.sam_prompt or name)
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

    def sam_aliases(self) -> dict[str, str]:
        """SAM prompt -> actor name, for objects whose prompt differs from their name."""
        return {s.sam_prompt: n for n, s in self.u.specs.items() if s.sam_prompt}

    def execute(self, action: Action) -> SkillResult:
        self._actions.log = []
        try:
            result = self._execute(action)
        finally:
            commands, self._actions.log = self._actions.log, None
        self.last_trace = (commands, result)
        return result

    def replay(self, trace) -> SkillResult:
        """Re-run the joint commands of an imagined execute() from the same state.
        CPU physics is deterministic, so the outcome is the one that was imagined."""
        commands, result = trace
        for a in commands:
            self.env.step(a)
        if commands:
            self.skills.planner.gripper_state = float(commands[-1][-1])
        return result

    def _execute(self, action: Action) -> SkillResult:
        s, objs = self.skills, self.u.objects
        t = action.type
        if t == ActionType.PICK:
            return s.pick(objs[action.target], action.grasp)
        if t == ActionType.PLACE_ON:
            ref = _np(objs[action.reference].pose.p[0])
            held = self._holding()
            if held is None:
                return SkillResult(False, "nothing in gripper")
            # Keep the object's current hang below the gripper; lower until its bottom clears the plate.
            hang = 0.0
            if held is not None:
                tcp_z = float(self.u.agent.tcp.pose.p[0, 2])
                bottom = float(objs[held].pose.p[0, 2]) - self.u.specs[held].half_height
                hang = tcp_z - bottom
            release_z = PLATE_TOP + PLACE_CLEARANCE + hang
            return s.place_on(np.array([ref[0], ref[1], release_z]), drop_height=0.0)
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
        if event.kind == "knock":   # physics decides whether a held object comes loose
            self._throw_ball(actor, np.array([event.dx, event.dy]))
        else:  # teleport: someone moved it while nobody was looking
            p, q = _np(actor.pose.p[0]), _np(actor.pose.q[0])
            actor.set_pose(sapien.Pose(p=[p[0] + event.dx, p[1] + event.dy,
                                          self.u.specs[event.object].half_height], q=q))
        self._hold(20)   # let it settle

    def _hold(self, steps: int) -> None:
        for _ in range(steps):
            self.env.step(_np(self.u.agent.robot.get_qpos()[:, :8]))

    def _throw_ball(self, actor, push_xy: np.ndarray) -> None:
        """Throw the chaos baseball at the object (held or not), travelling along push_xy.

        Elongated objects are hit near one end, perpendicular to their long axis,
        so a held banana twists out of the fingers. The launch velocity is solved
        so the ballistic arc passes through the aim point; whether the object comes
        loose is up to the physics.
        """
        import sapien
        ball = self.u.chaos_ball
        d = push_xy / (np.linalg.norm(push_xy) + 1e-9)
        aim = _np(actor.pose.p[0]).copy()
        if self.u.specs[self._name_of(actor)].kind == "banana":
            R = sapien.Pose(q=_np(actor.pose.q[0])).to_transformation_matrix()[:3, :3]
            long_axis = R[:, 1] * np.array([1.0, 1.0, 0.0])        # YCB banana is long along y
            if np.linalg.norm(long_axis) > 1e-3:
                long_axis /= np.linalg.norm(long_axis)
                aim[:2] += long_axis[:2] * KNOCK_END_OFFSET
                perp = np.array([-long_axis[1], long_axis[0]])
                d = perp if perp @ d >= 0 else -perp
        aim[2] = max(aim[2], CHAOS_BALL_RADIUS + 0.003)
        start = aim - np.array([d[0], d[1], 0.0]) * THROW_DISTANCE + np.array([0.0, 0.0, THROW_RISE])
        g = np.array([0.0, 0.0, -9.81])
        v0 = (aim - start) / THROW_TIME - 0.5 * g * THROW_TIME
        for body in ball._bodies:
            body.disable_gravity = False
        ball.set_pose(sapien.Pose(p=start))
        ball.set_angular_velocity(np.zeros((1, 3)))
        ball.set_linear_velocity(v0[None])
        self._hold(int(THROW_TIME / self.u.control_timestep) + 15)   # flight + aftermath
        self._park_ball()

    def _park_ball(self) -> None:
        import sapien
        ball = self.u.chaos_ball
        for body in ball._bodies:
            body.disable_gravity = True
        ball.set_linear_velocity(np.zeros((1, 3)))
        ball.set_angular_velocity(np.zeros((1, 3)))
        ball.set_pose(sapien.Pose(p=list(CHAOS_BALL_PARK)))

    def _name_of(self, actor) -> str:
        return next(n for n, a in self.u.objects.items() if a is actor)

    def is_goal_satisfied(self, goal: Goal) -> bool:
        return all(c["ok"] for c in self.goal_report(goal).values())

    def goal_report(self, goal: Goal) -> dict:
        objs = self.u.objects
        obj, ref = objs.get(goal.target), objs.get(goal.reference)
        if obj is None or ref is None:
            return {"objects_exist": {"ok": False, "value": [goal.target, goal.reference]}}
        po, pr = _np(obj.pose.p[0]), _np(ref.pose.p[0])
        dist = float(np.linalg.norm(po[:2] - pr[:2]))
        # Not "above the rim": the plate is concave, so a banana lying in it sits below rim height.
        min_z = self.u.specs[goal.target].half_height + ON_PLATE_LIFT
        speed = float(np.linalg.norm(_np(obj.linear_velocity[0])))
        report = {
            "released": {"ok": self._holding() != goal.target, "value": self._holding()},
            "over_reference": {"ok": dist < PLATE_RADIUS * 0.8, "value_cm": round(dist * 100, 1),
                               "need_below_cm": round(PLATE_RADIUS * 80, 1)},
            "above_table": {"ok": float(po[2]) > min_z, "value_cm": round(float(po[2]) * 100, 1),
                            "need_above_cm": round(min_z * 100, 1)},
            "at_rest": {"ok": speed < 0.05, "value_m_s": round(speed, 3)},
        }
        if self.u.specs[goal.target].kind in UPRIGHT_KINDS:
            import sapien
            up = sapien.Pose(q=_np(obj.pose.q[0])).to_transformation_matrix()[:3, 2]
            tilt = float(np.degrees(np.arccos(min(abs(up[2]), 1.0))))
            report["upright"] = {"ok": tilt < MAX_TILT_DEG, "tilt_deg": round(tilt, 1),
                                 "need_below_deg": MAX_TILT_DEG}
        return report

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
