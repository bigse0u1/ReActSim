"""The World interface: everything the agent loop needs from a simulator."""

from __future__ import annotations

from typing import Any, Protocol

from reactsim.types import Action, Goal, Observation, PerturbEvent, SkillResult


class World(Protocol):
    def reset(self, seed: int) -> None: ...

    def observe(self) -> Observation: ...

    def execute(self, action: Action) -> SkillResult: ...

    def perturb(self, event: PerturbEvent) -> None: ...

    def is_goal_satisfied(self, goal: Goal) -> bool:
        """Physics-based success check. Never delegated to an LLM."""
        ...

    # Imagination needs to fork physics and come back.
    def snapshot(self) -> Any: ...

    def restore(self, snap: Any) -> None: ...

    def frame(self) -> Any:
        """Rendered frame for video / dashboard, or None."""
        ...
