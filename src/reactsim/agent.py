"""The ReAct loop: perceive -> plan -> (imagine) -> act -> diagnose -> replan."""

from __future__ import annotations

import dataclasses
import time

from reactsim.chaos.perturb import Chaos
from reactsim.evaluation.diagnosis import diagnose
from reactsim.evaluation.metrics import EpisodeRecord
from reactsim.imagination.rollout import Imagination
from reactsim.perception.grounding import Perception
from reactsim.planning.schema import PlanValidationError
from reactsim.types import GRASP_MODES, Action, ActionType, FailureReport, FailureType
from reactsim.viz.dashboard import Dashboard


MAX_SAME_FAILURE = 3   # stop after the same failure follows the same action this many times


def _goal_not_met(world, goal, why: str) -> FailureReport:
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


class ReActAgent:
    def __init__(self, world, perception: Perception, planner, *,
                 imagination: Imagination | None = None, chaos: Chaos | None = None,
                 dashboard: Dashboard | None = None, max_steps: int = 20, max_replans: int = 4,
                 planner_name: str = "", world_name: str = ""):
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

    def _candidates(self, action: Action, alternatives: list[Action]) -> list[Action]:
        if action.type != ActionType.PICK:
            return [action]
        cands = [action] + [a for a in alternatives if a.type == ActionType.PICK]
        for g in GRASP_MODES:
            cands.append(dataclasses.replace(action, grasp=g, rationale="grasp variant"))
        seen, out = set(), []
        for c in cands:
            key = (c.type, c.target, c.grasp)
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    def run(self, instruction: str, seed: int) -> EpisodeRecord:
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
        goal, queue = plan.goal, list(plan.actions)
        self.dash.text(step, "plan", f"{[a.short() for a in queue]} — {plan.rationale}")

        replans = 0
        repeats: dict[tuple, int] = {}
        last_action = None
        while step < self.max_steps:
            if self.world.is_goal_satisfied(goal):
                rec.success = True
                self.dash.text(step, "success", f"{goal.target} {goal.relation} {goal.reference}")
                break

            failure = None
            if not queue:
                failure = _goal_not_met(self.world, goal, "plan finished but the success check fails")
            else:
                action = queue.pop(0)
                if action.type == ActionType.GIVE_UP:
                    rec.stop_reason = f"planner gave up: {action.rationale or plan.rationale}"
                    self.dash.text(step, "give_up", action.rationale or plan.rationale)
                    break
                if action.type == ActionType.DONE:
                    failure = _goal_not_met(self.world, goal, "planner said done but the success check fails")
                else:
                    if self.imagination is not None:
                        cands = self._candidates(action, alternatives)
                        if len(cands) > 1:
                            action, imagined = self.imagination.choose(cands, goal, step)
                            self.dash.imagined(step, imagined)
                    alternatives = []

                    last_action = action.short()
                    result = self.world.execute(action)
                    self.dash.text(step, "act", f"{action.short()} -> {result.message}")

                    for ev in self.chaos.events_for_step(step, list(state.objects)):
                        self.world.perturb(ev)
                        rec.perturbations += 1
                        self.dash.text(step, "chaos", f"{ev.source}: {ev.kind} {ev.object or ''}")

                    obs = self.world.observe()
                    after = self.perception.perceive(obs, step + 1)
                    self.dash.observation(step + 1, obs, self.perception.last_masks)
                    self.dash.objects(step + 1, after)
                    failure = diagnose(action, result, state, after, goal.target)
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
                if repeats[key] >= MAX_SAME_FAILURE:
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
                    plan, alternatives = self.planner.replan(instruction, goal, state, rec.steps, failure)
                except PlanValidationError as e:
                    self.dash.text(step, "planner", f"no valid recovery: {e}")
                    break
                queue = list(plan.actions)
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
