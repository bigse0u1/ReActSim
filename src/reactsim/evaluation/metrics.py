"""Episode records and aggregate metrics. Every number in the demo comes from these files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EpisodeRecord:
    seed: int
    planner: str
    world: str
    instruction: str
    success: bool = False
    initial_failure: bool = False
    recovery_attempts: int = 0
    perturbations: int = 0
    llm_calls: int = 0
    llm_latency_s: float = 0.0
    invalid_llm_outputs: int = 0
    wall_time_s: float = 0.0
    stop_reason: str = ""
    steps: list[dict] = field(default_factory=list)


def summarize(records: list[EpisodeRecord]) -> dict:
    n = len(records)
    failed_first = [r for r in records if r.initial_failure]
    return {
        "episodes": n,
        "task_success_rate": sum(r.success for r in records) / n if n else 0.0,
        "recovery_success_rate": (sum(r.success for r in failed_first) / len(failed_first)
                                  if failed_first else None),
        "mean_recovery_attempts": sum(r.recovery_attempts for r in records) / n if n else 0.0,
        "mean_llm_latency_s": sum(r.llm_latency_s for r in records) / n if n else 0.0,
    }


def save(records: list[EpisodeRecord], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r), default=str) + "\n")
    with open(out / "summary.json", "w") as f:
        json.dump(summarize(records), f, indent=2)
    return out
