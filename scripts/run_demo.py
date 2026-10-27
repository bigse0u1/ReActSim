"""Run one episode. Type perturbations on stdin during the run for Chaos Mode.

    python scripts/run_demo.py --offline                      # FakeWorld + mock Muse, no GPU/API
    python scripts/run_demo.py --world maniskill --offline    # real sim, mock planner
    python scripts/run_demo.py --world maniskill              # real sim + Muse Spark
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reactsim.build import build_agent, load_config  # noqa: E402
from reactsim.evaluation.metrics import save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--instruction", default="Put the banana on the plate")
    ap.add_argument("--world", choices=["fake", "maniskill"])
    ap.add_argument("--planner", choices=["muse", "rule", "fixed"])
    ap.add_argument("--segmenter", choices=["oracle", "sam3"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--offline", action="store_true", help="use the mock Muse client (no API calls)")
    ap.add_argument("--chaos-stdin", action="store_true", help="read live perturbations from stdin")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    cfg = load_config(args.config)
    if args.world:
        cfg["world"] = args.world
    if args.planner:
        cfg["planning"]["planner"] = args.planner
    if args.segmenter:
        cfg["perception"]["segmenter"] = args.segmenter
    seed = cfg["seed"] if args.seed is None else args.seed

    out = Path("outputs") / time.strftime("demo_%Y%m%d_%H%M%S")
    agent = build_agent(cfg, offline=args.offline, out_dir=str(out / "video"))

    if args.chaos_stdin:
        def reader():
            for line in sys.stdin:
                agent.chaos.submit(line.strip())
        threading.Thread(target=reader, daemon=True).start()
        print("Chaos Mode: type e.g. 'knock the banana' and press Enter.")

    try:
        rec = agent.run(args.instruction, seed)
    finally:
        close = getattr(agent.world, "close", None)
        if close:
            close()   # flushes the recorded video
    save([rec], out)
    print(f"\nsuccess={rec.success} recoveries={rec.recovery_attempts} "
          f"llm_calls={rec.llm_calls} -> {out}")


if __name__ == "__main__":
    main()
