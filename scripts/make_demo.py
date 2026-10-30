"""Record one episode as a demo video: the real run with captions, and each Imagination
step as its candidates playing side by side.

    python scripts/make_demo.py --chaos knock --seed 1               # Muse Spark, kitchen
    python scripts/make_demo.py --chaos drop --offline               # mock planner, no API
    python scripts/make_demo.py --scene home --robot hand --chaos drop --instruction "Put the can on the plate"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reactsim.build import build_agent, default_instruction, load_config  # noqa: E402
from reactsim.viz.demo_video import DemoRecorder  # noqa: E402

CHAOS = {"none": None,
         "knock": {"kind": "knock", "object": "target", "dx": -0.08, "dy": 0.0},
         "drop": {"kind": "drop"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scene", default="kitchen", choices=["kitchen", "home", "clutter"])
    ap.add_argument("--robot", default="panda", choices=["panda", "hand"])
    ap.add_argument("--chaos", default="knock", choices=list(CHAOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default=None)
    ap.add_argument("--segmenter", choices=["oracle", "sam3"], default="oracle")
    ap.add_argument("--grasp-from", choices=["sim", "perception"], default="sim")
    ap.add_argument("--offline", action="store_true", help="mock planner instead of Muse Spark")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(".env")
    cfg = load_config(args.config)
    cfg["world"] = "maniskill"
    cfg["maniskill"].update(scene=args.scene, robot=args.robot, record_video=False)
    cfg["perception"].update(segmenter=args.segmenter, grasp_from=args.grasp_from)
    cfg["planning"]["planner"] = "muse"
    cfg["chaos"]["enabled"] = CHAOS[args.chaos] is not None
    cfg["chaos"]["scripted"] = [{"step": 0, "event": CHAOS[args.chaos]}] if CHAOS[args.chaos] else []
    instruction = args.instruction or default_instruction(cfg)

    agent = build_agent(cfg, offline=args.offline)
    if agent.imagination is not None:
        agent.imagination.capture = True
    title = f'"{instruction}"'
    agent.dash = DemoRecorder(agent.world, title=title)
    out = Path("outputs") / time.strftime(f"demo_video_{args.chaos}_s{args.seed}_%Y%m%d_%H%M%S")
    try:
        rec = agent.run(instruction, args.seed)
        video = agent.dash.render(out / "demo.mp4")
    finally:
        agent.world.close()
    print(f"\nsuccess={rec.success} recoveries={rec.recovery_attempts} -> {video}")


if __name__ == "__main__":
    main()
