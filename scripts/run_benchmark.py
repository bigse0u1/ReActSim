"""Compare systems on identical seeds and perturbations.

    python scripts/run_benchmark.py --world fake --offline --seeds 20
    python scripts/run_benchmark.py --world maniskill --seeds 10      # real sim + Muse Spark

Systems: fixed (A), rule (B), muse without Imagination (ablation), muse+imagine (ours).
Conditions: normal (no disturbance), knock (chaos ball after the first action),
drop (gripper forced open after the first action).
Every episode is appended to <out>/<condition>/<system>/episodes.jsonl as it finishes,
so a long run can be stopped and still report what it has.
"""

import argparse
import copy
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reactsim.build import build_agent, default_instruction, load_config  # noqa: E402
from reactsim.evaluation.metrics import EpisodeRecord, summarize  # noqa: E402

CONDITIONS = {
    "normal": None,
    "knock": {"kind": "knock", "object": "target", "dx": -0.08, "dy": 0.0},
    "drop": {"kind": "drop"},
}
SYSTEMS = {   # name -> (planner, imagination)
    "fixed": ("fixed", False),
    "rule": ("rule", False),
    "muse": ("muse", False),
    "muse+imagine": ("muse", True),
}


def run_one(agent, instruction, seed, cond, system) -> EpisodeRecord:
    try:
        return agent.run(instruction, seed)
    except Exception as e:   # an API or planner error ends this episode, not the benchmark
        rec = EpisodeRecord(seed, system, cond, instruction)
        rec.stop_reason = f"error: {type(e).__name__}: {e}"[:300]
        traceback.print_exc()
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--world", choices=["fake", "maniskill"])
    ap.add_argument("--scene", choices=["kitchen", "home"])
    ap.add_argument("--robot", choices=["panda", "hand"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--instruction", default=None, help="default: put the scene's first object on the plate")
    ap.add_argument("--out", default=None, help="resume into an existing bench_* directory")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        pass
    base = load_config(args.config)
    if args.world:
        base["world"] = args.world
    if args.scene:
        base["maniskill"]["scene"] = args.scene
    if args.robot:
        base["maniskill"]["robot"] = args.robot
    base["maniskill"]["record_video"] = False
    instruction = args.instruction or default_instruction(base)
    root = Path(args.out) if args.out else Path("outputs") / time.strftime("bench_%Y%m%d_%H%M%S")

    table = {}
    for cond in args.conditions.split(","):
        for system in args.systems.split(","):
            planner, imagine = SYSTEMS[system]
            cfg = copy.deepcopy(base)
            cfg["planning"]["planner"] = planner
            cfg["imagination"]["enabled"] = imagine
            cfg["chaos"]["enabled"] = CONDITIONS[cond] is not None
            cfg["chaos"]["scripted"] = [{"step": 0, "event": CONDITIONS[cond]}] if CONDITIONS[cond] else []
            out = root / cond / system
            out.mkdir(parents=True, exist_ok=True)
            log = out / "episodes.jsonl"
            done = {json.loads(l)["seed"]: EpisodeRecord(**json.loads(l)) for l in log.open()} if log.exists() else {}
            todo = [s for s in range(args.seeds) if s not in done]
            if todo:
                agent = build_agent(cfg, offline=args.offline)
                agent.dash.text = lambda *a, **k: None   # quiet
                try:
                    for s in todo:
                        t0 = time.time()
                        rec = run_one(agent, instruction, s, cond, system)
                        rec.planner = system
                        done[s] = rec
                        with log.open("a") as f:
                            f.write(json.dumps(asdict(rec), default=str) + "\n")
                        print(f"{cond:7s} {system:13s} seed {s:2d}  success={rec.success!s:5s} "
                              f"recoveries={rec.recovery_attempts} llm={rec.llm_calls} "
                              f"{time.time() - t0:5.0f}s  {rec.stop_reason}", flush=True)
                finally:
                    close = getattr(agent.world, "close", None)
                    if close:
                        close()
            recs = [done[s] for s in sorted(done) if s < args.seeds]
            s = summarize(recs)
            s["mean_llm_calls"] = sum(r.llm_calls for r in recs) / len(recs) if recs else 0.0
            s["errors"] = sum(r.stop_reason.startswith("error") for r in recs)
            table.setdefault(cond, {})[system] = s
            (root / "table.json").write_text(json.dumps(table, indent=2))

    lines = ["| condition | system | success | recovered after a failure | mean recoveries | LLM calls | errors |",
             "|---|---|---|---|---|---|---|"]
    for cond, rows in table.items():
        for system, s in rows.items():
            rec = "-" if s["recovery_success_rate"] is None else f"{s['recovery_success_rate']:.0%}"
            lines.append(f"| {cond} | {system} | {s['task_success_rate']:.0%} ({s['episodes']}) | {rec} | "
                         f"{s['mean_recovery_attempts']:.1f} | {s['mean_llm_calls']:.1f} | {s['errors']} |")
    (root / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {root}")
    if base["world"] == "fake":
        print("\nNOTE: FakeWorld numbers validate the pipeline only. Do not report them as robot results.")


if __name__ == "__main__":
    main()
