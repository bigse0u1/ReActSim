"""Smoke test for the Meta Model API: one Muse Spark call, one SAM 3.1 call per prompt.

    python scripts/check_api.py                 # Muse + SAM on the sim camera
    python scripts/check_api.py --skip-sam

SAM masks are compared with the simulator's ground-truth segmentation (IoU) and
an overlay is written to outputs/check_api_sam.png.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check_muse(model: str) -> None:
    from reactsim.planning.llm import MetaChatClient

    client = MetaChatClient(model, temperature=0.0)
    text = client.chat("Reply with exactly one word.", "Say OK.")
    print(f"[muse] {model}: {text.strip()[:80]!r}  ({client.calls[-1].latency_s:.1f}s, usage={client.calls[-1].usage})")


def check_sam(model: str, prompts: list[str]) -> None:
    import numpy as np
    from PIL import Image

    from reactsim.perception.segmenter import OracleSegmenter, Sam3Segmenter
    from reactsim.worlds.maniskill import ManiSkillWorld

    world = ManiSkillWorld(record_dir=None)
    world.reset(0)
    obs = world.observe()
    oracle = OracleSegmenter(world.segmentation_ids(), world.sam_aliases())
    oracle.segmentation = obs.segmentation
    gt = oracle.segment(obs.rgb, prompts)

    sam = Sam3Segmenter(model)
    masks = sam.segment(obs.rgb, prompts)
    overlay = obs.rgb.astype(float)
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255)]
    for i, p in enumerate(prompts):
        m = masks.get(p)
        if m is None:
            print(f"[sam]  {p:<8} no mask")
            continue
        g = gt.get(p)
        iou = (m.mask & g.mask).sum() / max((m.mask | g.mask).sum(), 1) if g else float("nan")
        print(f"[sam]  {p:<8} pixels={int(m.mask.sum()):>6} conf={m.score:.2f} IoU_vs_GT={iou:.2f}  ({sam.calls[i]:.1f}s)")
        overlay[m.mask] = 0.5 * overlay[m.mask] + 0.5 * np.array(colors[i % len(colors)])
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    Image.fromarray(np.hstack([obs.rgb, overlay.astype(np.uint8)])).save(out / "check_api_sam.png")
    print(f"[sam]  overlay -> {out / 'check_api_sam.png'}")
    world.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse-model", default="muse-spark-1.3")
    ap.add_argument("--sam-model", default="sam-3.1")
    ap.add_argument("--prompts", default="banana,cup,can,plate")
    ap.add_argument("--skip-muse", action="store_true")
    ap.add_argument("--skip-sam", action="store_true")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not args.skip_muse:
        check_muse(args.muse_model)
    if not args.skip_sam:
        check_sam(args.sam_model, args.prompts.split(","))


if __name__ == "__main__":
    main()
