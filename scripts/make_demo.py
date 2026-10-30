"""Record one episode as a demo video: the real run with captions, and each Imagination
step as its candidates playing side by side.

    python scripts/make_demo.py --chaos knock --seed 1               # Muse Spark, kitchen
    python scripts/make_demo.py --chaos drop --offline               # mock planner, no API
    python scripts/make_demo.py --scene home --robot hand --chaos drop --instruction "Put the can on the plate"
    python scripts/make_demo.py --say "knock it out of the robot's hand" --chaos none   # natural-language Chaos
    python scripts/make_demo.py --scene tidy --chaos undo --instruction "Put the fruit on the plate and the can in the bowl"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reactsim.build import build_agent, default_instruction, load_config  # noqa: E402
from reactsim.viz.demo_video import DemoRecorder  # noqa: E402

CHAOS = {"none": None,
         "knock": (0, {"kind": "knock", "object": "target", "dx": -0.08, "dy": 0.0}),
         "drop": (0, {"kind": "drop"}),
         # several-goal tasks: after the second place, someone takes the banana back off the plate
         "undo": (3, {"kind": "teleport", "object": "banana", "dx": 0.0, "dy": -0.25})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scene", default="kitchen", choices=["kitchen", "home", "clutter", "tidy"])
    ap.add_argument("--robot", default="panda", choices=["panda", "hand"])
    ap.add_argument("--chaos", default="knock", choices=list(CHAOS))
    ap.add_argument("--say", default=None,
                    help='a spectator\'s sentence Muse Spark turns into the disturbance, e.g. "knock it out of its hand"')
    ap.add_argument("--say-step", type=int, default=0, help="after which step the sentence is said")
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
    chaos = CHAOS[args.chaos]
    cfg["chaos"]["enabled"] = chaos is not None
    cfg["chaos"]["scripted"] = [{"step": chaos[0], "event": chaos[1]}] if chaos else []
    if args.say:   # the natural-language path instead of a scripted event
        cfg["chaos"].update(enabled=True, scripted=[{"step": args.say_step, "say": args.say}])
    instruction = args.instruction or default_instruction(cfg)

    agent = build_agent(cfg, offline=args.offline)
    if agent.imagination is not None:
        agent.imagination.capture = True
    title = f'"{instruction}"'
    agent.dash = DemoRecorder(agent.world, title=title)
    out = Path("outputs") / time.strftime(f"demo_video_{'say' if args.say else args.chaos}_s{args.seed}_%Y%m%d_%H%M%S")
    try:
        rec = agent.run(instruction, args.seed)
        video = agent.dash.render(out / "demo.mp4")
    finally:
        agent.world.close()
    print(f"\nsuccess={rec.success} recoveries={rec.recovery_attempts} -> {video}")


if __name__ == "__main__":
    main()
