"""Show, don't tell: learn a task from a phone video of a human hand, then let the robot do it.

    python scripts/learn_from_video.py data/demos/can_side.mp4 --watch-only   # video -> demonstration.json
    python scripts/learn_from_video.py data/demos/can_side.mp4                # ... then run the robot
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reactsim.build import load_config  # noqa: E402
from reactsim.envs.tabletop import SCENES  # noqa: E402
from reactsim.imitation.demonstration import BODIES, Demonstration, learn_from_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--scene", default="home", choices=list(SCENES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot", choices=["panda", "hand"], default="panda")
    ap.add_argument("--watch-only", action="store_true", help="stop after understanding the video")
    ap.add_argument("--demo", help="reuse a saved demonstration.json instead of calling SAM/Muse again")
    ap.add_argument("--segmenter", choices=["oracle", "sam3"], help="robot-side perception")
    ap.add_argument("--offline", action="store_true", help="mock Muse for the robot run")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(".env")
    cfg = load_config(args.config)
    out = Path("outputs") / time.strftime(f"imitate_{Path(args.video).stem}_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        demo = Demonstration.load(args.demo)
    else:
        from reactsim.perception.video import SamVideoTracker
        from reactsim.planning.llm import MetaChatClient
        muse = MetaChatClient(cfg["planning"]["model"], temperature=0.0)
        tracker = SamVideoTracker(cfg["perception"]["sam_model"])
        robot_objects = list(SCENES[args.scene])
        demo = learn_from_video(args.video, muse, tracker, robot_objects, out)
        print(f"Muse calls: {len(muse.calls)} ({sum(c.latency_s for c in muse.calls):.1f}s), "
              f"SAM frames billed: {tracker.frames_billed}")
    demo.save(out / "demonstration.json")
    print(f"watched  {demo.task}")
    print(f"         moved '{demo.target}' onto '{demo.reference}' "
          f"(grasp at frame {demo.grasp_frame}, released at {demo.release_frame})")
    print(f"         human grasp: {demo.human_grasp} — {demo.grasp_evidence}")
    print(f"robot    {demo.robot_target} -> {demo.robot_reference}")
    print(f"         instruction: {demo.instruction(BODIES[args.robot])}")
    if args.watch_only:
        print(f"-> {out}")
        return
    if not (demo.robot_target and demo.robot_reference):
        sys.exit(f"the demonstrated objects have no counterpart in the '{args.scene}' scene")

    from reactsim.build import build_agent
    from reactsim.evaluation.metrics import save
    cfg["world"] = "maniskill"
    cfg["maniskill"]["scene"] = args.scene
    cfg["maniskill"]["robot"] = args.robot
    cfg["planning"]["planner"] = "muse"
    cfg["chaos"]["enabled"] = False
    if args.segmenter:
        cfg["perception"]["segmenter"] = args.segmenter
    agent = build_agent(cfg, offline=args.offline, out_dir=str(out / "video"))
    try:
        rec = agent.run(demo.instruction(BODIES[args.robot]), args.seed)
    finally:
        agent.world.close()
    save([rec], out)
    print(adaptation(demo, rec, BODIES[args.robot]))
    if demo.sam_overlay and Path(demo.sam_overlay).exists():
        from reactsim.imitation.demonstration import SAMPLE_FPS
        from reactsim.viz.imitation_video import compose
        robot_video = sorted((out / "video").glob("*.mp4"))[-1]
        video = compose(demo.sam_overlay, robot_video, demo, rec.imagined, used_grasp(rec),
                        rec.success, out / "show_dont_tell.mp4", SAMPLE_FPS,
                        body={"panda": "two-finger gripper", "hand": "five-finger hand"}[args.robot])
        print(f"video    {video}")
    print(f"\nsuccess={rec.success} recoveries={rec.recovery_attempts} -> {out}")


def used_grasp(rec) -> str | None:
    picks = [s["action"] for s in rec.steps if s["action"].startswith("pick(")]
    return picks[-1].split("[")[-1].rstrip("]") if picks else None


def adaptation(demo: Demonstration, rec, body: str) -> str:
    """How the robot's grasp relates to the human's, from what Imagination tried."""
    used, human = used_grasp(rec), demo.human_grasp
    if used is None:
        return "robot    never picked"
    lines = [f"robot    grasp used: {used} (human: {human})"]
    for i in rec.imagined:
        for c in i["candidates"]:
            lines.append(f"  [{i['step']:02d}]  imagined {c['action']:<24} score={c['score']:+.2f}  {c['failure']}")
    tried = [c for i in rec.imagined for c in i["candidates"] if c["action"].endswith(f"[{human}]")]
    failed = [c for c in tried if c["failure"] != "none"]
    if used == human:
        lines.append(f"         -> the human's {human} grasp works for {body} too")
    elif failed:
        lines.append(f"         -> in imagination the human's {human} grasp ended in {failed[-1]['failure']} "
                     f"for {body} ({failed[-1].get('detail', '')}), so the robot grasped {used} instead")
    else:
        lines.append(f"         -> the robot grasped {used}; the human's {human} grasp was not needed")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
