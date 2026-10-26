"""Compare fixed / rule / muse(+imagination) on identical seeds and perturbations.

    python scripts/run_benchmark.py --world fake --offline --seeds 20
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reactsim.build import build_agent, load_config  # noqa: E402
from reactsim.evaluation.metrics import save, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--world", choices=["fake", "maniskill"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--planners", default="fixed,rule,muse")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--instruction", default="Put the banana on the plate")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    base = load_config(args.config)
    if args.world:
        base["world"] = args.world
    root = Path("outputs") / time.strftime("bench_%Y%m%d_%H%M%S")
    table = {}
    for name in args.planners.split(","):
        cfg = copy.deepcopy(base)
        cfg["planning"]["planner"] = name
        agent = build_agent(cfg, offline=args.offline)
        agent.dash.text = lambda *a, **k: None   # quiet
        recs = [agent.run(args.instruction, s) for s in range(args.seeds)]
        save(recs, root / name)
        table[agent.planner_name] = summarize(recs)
    (root / "table.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))
    if base["world"] == "fake":
        print("\nNOTE: FakeWorld numbers validate the pipeline only. Do not report them as robot results.")


if __name__ == "__main__":
    main()
