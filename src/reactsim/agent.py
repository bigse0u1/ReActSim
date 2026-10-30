"""The ReAct loop: perceive -> plan -> (imagine) -> act -> diagnose -> replan."""

from __future__ import annotations

import dataclasses
import time

from reactsim.chaos.perturb import Chaos
from reactsim.evaluation.diagnosis import diagnose, goals_not_met
from reactsim.evaluation.metrics import EpisodeRecord
from reactsim.imagination.rollout import Imagination
from reactsim.perception.grounding import Perception
from reactsim.planning.schema import PlanValidationError
from reactsim.types import GRASP_MODES, Action, ActionType, FailureReport, FailureType, Goal
from reactsim.viz.dashboard import Dashboard


def _goal_str(g: Goal) -> str:
    return f"{g.target} {g.relation} {g.reference}"


MAX_SAME_FAILURE = 3   # stop after the same failure follows the same action this many times
MAX_NO_WORKING_GRASP = 2   # ... sooner when Imagination already showed nothing works


class ReActAgent:
    def __init__(self, world, perception: Perception, planner, *,
                 imagination: Imagination | None = None, chaos: Chaos | None = None,
                 dashboard: Dashboard | None = None, max_steps: int = 20, max_replans: int = 4,
                 planner_name: str = "", world_name: str = "", approaches=(0,)):
        self.world = world
        self.perception = perception
        self.planner = planner
        self.imagination = imagination
        self.chaos = chaos or Chaos()
        self.dash = dashboard or Dashboard(enabled=False)
        self.max_steps = max_steps
        self.max_replans = max_replans
        self.planner_name = planner_name or type(planner).__name__
        self.world_name = world_name or type(world).__name__
        self.approaches = tuple(approaches)   # sides Imagination tries for non-top grasps (deg)

    def _candidates(self, action: Action, alternatives: list[Action]) -> list[Action]:
        if action.type != ActionType.PICK:
            return [action]
        cands = [action] + [a for a in alternatives if a.type == ActionType.PICK]
        for g in GRASP_MODES:
            for deg in ((0,) if g == "top" else self.approaches):
                cands.append(dataclasses.replace(action, grasp=g, approach_deg=deg or None,
                                                 rationale="grasp variant"))
        seen, out = set(), []
        for c in cands:
            key = (c.type, c.target, c.grasp, c.approach_deg or 0)
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def run(self, instruction: str, seed: int, goals: list[Goal] | None = None) -> EpisodeRecord:
        """goals: the task's own goals, when known (benchmarks). Success is judged on them, and
        the goals the planner read from the instruction are recorded next to them. Without
        them the planner's goals define the task."""
        rec = EpisodeRecord(seed, self.planner_name, self.world_name, instruction)
        t_start = time.time()
        self._calls_before = len(self._llm_calls())
        self.world.reset(seed)
        step = 0
        state = self._look(step)

        try:
            plan, alternatives = self.planner.plan(instruction, state, rec.steps)
        except PlanValidationError as e:
            self.dash.text(step, "planner", f"no valid plan: {e}")
            return self._finish(rec, t_start)
        goals = list(goals or plan.goals)
        rec.goals = [_goal_str(g) for g in goals]
        rec.planned_goals = [_goal_str(g) for g in plan.goals]
        queue = list(plan.actions)
        self.dash.text(step, "plan", f"{[a.short() for a in queue]} — {plan.rationale}")

        def unmet():
            return [g for g in goals if not self.world.is_goal_satisfied(g)]

        def focus(action):
            """The goal the current action works toward: the one for the object it picks (or
            the object in hand), else the first one still open."""
            name = action.target if action.type == ActionType.PICK else state.holding
            return next((g for g in goals if g.target == name), (unmet() or goals)[0])

        replans = 0
        held_before: list[Goal] = []   # goals that held after the previous step
        pending_replay = None   # (follow-up action, its imagined trace)
        repeats: dict[tuple, int] = {}
        last_action = None
        while step < self.max_steps:
            open_goals = unmet()
            held_before = [g for g in goals if g not in open_goals]
            if not open_goals:
                rec.success = True
                self.dash.text(step, "success", "; ".join(_goal_str(g) for g in goals))
                break

            failure = None
            if not queue:
                failure = goals_not_met(self.world, open_goals, "plan finished but the success check fails")
            else:
                action = queue.pop(0)
                if action.type == ActionType.GIVE_UP:
                    rec.stop_reason = f"planner gave up: {action.rationale or plan.rationale}"
                    self.dash.text(step, "give_up", action.rationale or plan.rationale)
                    break
                if action.type == ActionType.DONE:
                    failure = goals_not_met(self.world, open_goals, "planner said done but the success check fails")
                else:
                    chosen, imagined = None, []
                    goal = focus(action)
                    if self.imagination is not None:
                        cands = self._candidates(action, alternatives)
                        if len(cands) > 1:
                            then = queue[0] if queue and queue[0].type == ActionType.PLACE_ON else None
                            action, imagined = self.imagination.choose(cands, goal, step, then)
                            chosen = next((i for i in imagined if i.action is action), None)
                            if chosen is not None and chosen.then_trace is not None:
                                pending_replay = (chosen.then, chosen.then_trace)
                            self.dash.imagined(step, imagined)
                            rec.imagined.append({
                                "step": step, "proposed": cands[0].short(), "chosen": action.short(),
                                "candidates": [{"action": i.action.short(), "score": round(i.score, 2),
                                                "failure": i.failure.type.value, "detail": i.failure.detail}
                                               for i in imagined]})
                    alternatives = []

                    last_action = action.short()
                    if imagined and all(i.failure.is_failure and i.failure.type != FailureType.OBJECT_MOVED
                                        for i in imagined):
                        # Every candidate fails in physics and none even moves the object, so
                        # executing one would fail the same way and change nothing. Report it.
                        # (A candidate that moves the object is still tried: it can nudge the
                        # object somewhere graspable.)
                        tried = rec.imagined[-1]["candidates"]
                        failure = FailureReport(
                            FailureType.NO_WORKING_GRASP,
                            f"no grasp works on {action.target} where it is now: imagined "
                            + "; ".join(f"{c['action']} -> {c['detail'] or c['failure']}" for c in tried),
                            {"imagined": tried})
                        rec.steps.append({"step": step, "action": f"imagine {action.short()}", "ok": False,
                                          "failure": failure.type.value, "detail": failure.detail})
                        last_action = f"imagine {action.target}"
                    else:
                        failure = None
                if failure is None and action.type not in (ActionType.DONE, ActionType.GIVE_UP):
                    replay = (chosen.trace if chosen is not None and chosen.trace is not None
                              else pending_replay[1] if pending_replay and pending_replay[0] is action
                              else None)
                    if chosen is None and replay is None:
                        pending_replay = None
                    if replay is not None and hasattr(self.world, "replay"):
                        result = self.world.replay(replay)   # do exactly what was imagined
                    else:
                        result = self.world.execute(action)
                    self.dash.text(step, "act", f"{action.short()} -> {result.message}")

                    for ev in self.chaos.events_for_step(step, list(state.objects)):
                        if ev.object == "target":
                            ev = dataclasses.replace(ev, object=goal.target)
                        self.world.perturb(ev)
                        pending_replay = None   # the imagined follow-up no longer applies
                        rec.perturbations += 1
                        self.dash.text(step, "chaos", f"{ev.source}: {ev.kind} {ev.object or ''}")

                    obs = self.world.observe()
                    after = self.perception.perceive(obs, step + 1)
                    self.dash.observation(step + 1, obs, self.perception.last_masks)
                    self.dash.objects(step + 1, after)
                    failure = diagnose(action, result, state, after, goal.target)
                    undone = [g for g in held_before if not self.world.is_goal_satisfied(g)]
                    if not failure.is_failure and undone:
                        # Something already done came undone (e.g. someone took it back off the
                        # plate): notice now, not when the rest of the plan runs out.
                        failure = goals_not_met(self.world, undone, "; ".join(_goal_str(g) for g in undone)
                                                + " held before and no longer does (something moved it)")
                    rec.steps.append({"step": step, "action": action.short(), "ok": result.ok,
                                      "failure": failure.type.value, "detail": failure.detail})
                    state = after
                    step += 1
                    if action.type == ActionType.REOBSERVE:
                        state = self._look(step)

            if failure is not None and failure.is_failure:
                rec.initial_failure = True
                key = (failure.type, last_action)
                repeats[key] = repeats.get(key, 0) + 1
                if repeats[key] > 1:
                    failure.evidence["repeat_count"] = repeats[key]
                    failure.detail += (f" [same failure after {last_action} {repeats[key]} times:"
                                       " the previous recovery did not work, change strategy]")
                self.dash.text(step, "failure", f"{failure.type.value}: {failure.detail}")
                limit = MAX_NO_WORKING_GRASP if failure.type == FailureType.NO_WORKING_GRASP else MAX_SAME_FAILURE
                if repeats[key] >= limit:
                    rec.stop_reason = f"gave up: {failure.type.value} repeated {repeats[key]} times"
                    self.dash.text(step, "give_up", rec.stop_reason)
                    break
                if replans >= self.max_replans:
                    rec.stop_reason = "replan budget exhausted"
                    break
                replans += 1
                rec.recovery_attempts = replans
                state = self._look(step)   # look again before deciding how to recover
                try:
                    achieved = [g for g in goals if g not in unmet()]
                    plan, alternatives = self.planner.replan(instruction, unmet() or goals, state, rec.steps,
                                                             failure, achieved)
                except PlanValidationError as e:
                    self.dash.text(step, "planner", f"no valid recovery: {e}")
                    break
                queue = list(plan.actions)
                pending_replay = None
                self.dash.text(step, "replan", f"{[a.short() for a in queue]} — {plan.rationale}")

        return self._finish(rec, t_start)

    def _look(self, step: int):
        """Perceive with the segmenter (SAM 3.1) and log what was seen."""
        obs = self.world.observe()
        state = self.perception.perceive(obs, step, look=True)
        self.dash.observation(step, obs, self.perception.last_masks)
        self.dash.objects(step, state)
        if self.perception.last_masks or self.perception.last_errors:
            seen = ", ".join(
                f"{o.name}" + (f" ({self.perception.last_errors[o.name] * 100:.1f} cm off)"
                               if o.name in self.perception.last_errors else "")
                for o in state.objects.values() if o.confidence > 0)
            unseen = [o.name for o in state.objects.values() if o.confidence == 0]
            self.dash.text(step, "see", seen + (f" | not found: {', '.join(unseen)}" if unseen else ""))
        return state

    def _llm_calls(self) -> list:
        return getattr(getattr(self.planner, "client", None), "calls", [])

    def _finish(self, rec: EpisodeRecord, t_start: float) -> EpisodeRecord:
        rec.wall_time_s = time.time() - t_start
        calls = self._llm_calls()[self._calls_before:]
        rec.llm_calls = len(calls)
        rec.llm_latency_s = sum(c.latency_s for c in calls)
        rec.invalid_llm_outputs = getattr(self.planner, "invalid_outputs", 0)
        return rec
