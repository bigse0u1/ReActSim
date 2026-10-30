"""Physics-grounded failure detection: compare what an action should have done with what happened."""

from __future__ import annotations

import numpy as np

from reactsim.types import Action, ActionType, FailureReport, FailureType, SkillResult, WorldState

MOVED_EPS = 0.03   # m an untouched object may drift before we call it "moved"


def goal_not_met(world, goal, why: str) -> FailureReport:
    """Explain which success condition failed, with the measured values."""
    report = world.goal_report(goal)
    unmet = {k: v for k, v in report.items() if not v["ok"]}
    parts = []
    for name, c in unmet.items():
        vals = ", ".join(f"{k}={v}" for k, v in c.items() if k != "ok")
        parts.append(f"{name} ({vals})" if vals else name)
    return FailureReport(FailureType.GOAL_NOT_MET,
                         f"{why}; unmet: {'; '.join(parts) or 'none'}",
                         {"success_check": report})


def goals_not_met(world, goals, why: str) -> FailureReport:
    """goal_not_met for several goals: each open goal with the conditions it fails."""
    if len(goals) == 1:
        return goal_not_met(world, goals[0], why)
    parts, reports = [], {}
    for g in goals:
        report = world.goal_report(g)
        reports[f"{g.target} {g.relation} {g.reference}"] = report
        bad = [f"{k} ({', '.join(f'{a}={b}' for a, b in c.items() if a != 'ok')})"
               for k, c in report.items() if not c["ok"]]
        parts.append(f"{g.target} {g.relation} {g.reference}: " + ("; ".join(bad) or "none"))
    return FailureReport(FailureType.GOAL_NOT_MET, f"{why}; open goals: " + " | ".join(parts),
                         {"success_check": reports})


def diagnose(action: Action, result: SkillResult, before: WorldState,
             after: WorldState, goal_target: str) -> FailureReport:
    if not result.ok:
        return FailureReport(FailureType.CONTROLLER_ERROR, result.message)

    t = action.type
    if t == ActionType.PICK and after.holding != action.target:
        obj, prev = after.objects.get(action.target), before.objects.get(action.target)
        if obj is not None and prev is not None:
            shift = float(np.linalg.norm(obj.position[:2] - prev.position[:2]))
            if shift > MOVED_EPS:
                return FailureReport(FailureType.OBJECT_MOVED,
                                     f"{action.target} moved {shift * 100:.1f} cm during the grasp",
                                     {"from": prev.position.tolist(), "to": obj.position.tolist()})
        return FailureReport(
            FailureType.GRASP_FAILED,
            f"pick({action.target}, grasp={action.grasp}) finished but the object is not in the gripper",
            {"object_z": float(obj.position[2]) if obj else None, "grasp": action.grasp},
        )
    if t == ActionType.PLACE_ON and after.holding is None and before.holding is not None:
        obj, ref = after.objects[before.holding], after.objects[action.reference]
        if np.any(np.abs(obj.position[:2] - ref.position[:2]) > 0.06):
            return FailureReport(FailureType.PLACE_MISSED,
                                 f"{obj.name} landed off {ref.name}",
                                 {"offset_xy": (obj.position[:2] - ref.position[:2]).tolist()})

    # Held object lost without a place action.
    if before.holding and after.holding is None and t != ActionType.PLACE_ON:
        return FailureReport(FailureType.DROPPED, f"{before.holding} slipped out of the gripper")

    # Goal object moved by something other than the robot.
    b, a = before.objects.get(goal_target), after.objects.get(goal_target)
    if b is not None and a is not None and not b.held and not a.held and t != ActionType.PUSH_ASIDE:
        shift = float(np.linalg.norm(a.position[:2] - b.position[:2]))
        if shift > MOVED_EPS and t != ActionType.PLACE_ON:
            return FailureReport(FailureType.OBJECT_MOVED,
                                 f"{goal_target} moved {shift * 100:.1f} cm unexpectedly",
                                 {"from": b.position.tolist(), "to": a.position.tolist()})
    if a is not None and a.confidence == 0.0:
        return FailureReport(FailureType.TARGET_LOST, f"{goal_target} not visible to the camera")
    return FailureReport(FailureType.NONE)
