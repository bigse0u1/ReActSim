"""Stitch the recorded demo clips into one ~3 minute video with title, section and results cards.

    python scripts/make_final_video.py                       # -> outputs/final_video.mp4

Clips are recorded separately (scripts/make_demo.py, scripts/learn_from_video.py); edit CLIPS
to swap one. Busy clips are sped up by dropping frames; RESULTS is the closing card.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 20
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, WHITE, GREY, ACCENT, GOOD = (18, 20, 24), (235, 235, 235), (150, 155, 165), (255, 175, 70), (110, 220, 130)

# (section title, one line under it, clip, speed)
CLIPS = [
    ("Chaos: a baseball knocks the banana out of the gripper",
     "The robot is not told. It notices, explains the failure, and replans.",
     "outputs/demo_video_knock_s1_20261029_120148/demo.mp4", 1.5),
    ("The LLM proposes, physics disposes",
     "Muse Spark guesses a grasp; Imagination tries every grasp in a copy of the physics first.",
     "outputs/demo_video_drop_s1_20261029_120241/demo.mp4", 2.0),
    ("Going around",
     "Here the top and robot-side grasps fail; Imagination finds one from the robot's left.",
     "outputs/demo_video_none_s2_20261029_115139/demo.mp4", 1.0),
    ("Anyone can make trouble, in plain words",
     "A spectator types a sentence; Muse Spark turns it into a physical disturbance.",
     "outputs/demo_video_say_s1_20261030_204636/demo.mp4", 1.25),
    ("Several steps: \"Put the fruit on the plate and the can in the bowl\"",
     "Muse Spark grounds \"the fruit\". Someone takes the banana back off the plate; the robot notices at once.",
     "outputs/demo_video_undo_s0_20261030_182537/demo.mp4", 3.0),
    ("Same pre-check, a different body",
     "From a phone video: SAM 3.1 tracks the hand and the orange, Muse Spark reads the task, "
     "and the pre-check confirms the human's grasp also works for a five-finger hand.",
     "outputs/imitate_orange_top_20261030_234212/show_dont_tell.mp4", 1.0),
]

TITLE = ("ReActSim", "A robot that checks before it acts, and recovers when the world fights back",
         "Muse Spark plans  ·  SAM 3.1 sees  ·  ManiSkill 3 physics imagines")
PROBLEM = ["An LLM can plan a robot task from one sentence.",
           "But a plan made once breaks when a grasp slips or someone knocks the object away.",
           "ReActSim closes the loop: see, plan, imagine, act, diagnose, replan."]
# Real ManiSkill runs, real Muse Spark, 20 seeds unless noted (Fisher exact, one-sided).
RESULTS = [
    ("From SAM 3.1 perception only (banana, drop + knock)", "40/40", "28/40 Muse alone · 34/40 rule-based retry", "p = 0.0002"),
    ("Six object × disturbance cells, current defaults", "116/120", "105/120 rule-based retry", "p = 0.015"),
    ("Three goals, two containers (10 seeds × 3 conditions)", "28/30", "18/30 rule-based retry", "p = 0.005"),
    ("Can knocked over, stood back up and placed", "19/20", "16/20 rule-based retry · 0/9 before", "1.2 vs 1.8 recoveries"),
    ("Cluttered table, grasps from every side", "19/20", "17/20 robot side only · 15/20 rule-based", "0.35 vs 1.30 recoveries"),
]


def fonts():
    return ({s: ImageFont.truetype(FONT, s) for s in (22, 26, 30)},
            {s: ImageFont.truetype(FONT_BOLD, s) for s in (24, 30, 36, 44, 80)})


def card(draw_fn, seconds: float) -> list[np.ndarray]:
    im = Image.new("RGB", (W, H), BG)
    draw_fn(ImageDraw.Draw(im))
    frame = np.asarray(im)
    return [frame] * int(seconds * FPS)


def wrap(draw, text, font, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=font) > width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    return lines + [line]


def clip_frames(path: str, speed: float) -> list[np.ndarray]:
    r = imageio.get_reader(path)
    src_fps = r.get_meta_data()["fps"]
    frames = [f for f in r]
    step = speed * src_fps / FPS                 # source frames per output frame
    out, t = [], 0.0
    while t < len(frames):
        f = frames[int(t)]
        if f.shape[:2] != (H, W):
            f = np.asarray(Image.fromarray(f).resize((W, H)))
        out.append(f)
        t += step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/final_video.mp4")
    args = ap.parse_args()
    reg, bold = fonts()
    frames: list[np.ndarray] = []

    def title(d):
        d.text((W // 2, 250), TITLE[0], font=bold[80], fill=WHITE, anchor="mm")
        d.text((W // 2, 350), TITLE[1], font=reg[30], fill=WHITE, anchor="mm")
        d.text((W // 2, 420), TITLE[2], font=reg[26], fill=ACCENT, anchor="mm")
    frames += card(title, 3.5)

    def problem(d):
        y = 240
        for i, text in enumerate(PROBLEM):
            font = bold[30] if i == 2 else reg[30]
            for line in wrap(d, text, font, W - 160):
                d.text((W // 2, y), line, font=font, fill=ACCENT if i == 2 else WHITE, anchor="mm")
                y += 44
            y += 26
    frames += card(problem, 4.5)

    for i, (head, sub, path, speed) in enumerate(CLIPS, 1):
        if not Path(path).exists():
            sys.exit(f"missing clip: {path}")

        def section(d, head=head, sub=sub, i=i):
            d.text((W // 2, 280), f"{i} / {len(CLIPS)}", font=reg[26], fill=GREY, anchor="mm")
            d.text((W // 2, 340), head, font=bold[36], fill=WHITE, anchor="mm")
            y = 410
            for line in wrap(d, sub, reg[26], W - 200):
                d.text((W // 2, y), line, font=reg[26], fill=ACCENT, anchor="mm")
                y += 38
            if speed > 1:
                d.text((W - 30, H - 30), f"played at {speed:g}x", font=reg[22], fill=GREY, anchor="rb")
        frames += card(section, 2.0)
        frames += clip_frames(path, speed)

    def results(d):
        d.text((W // 2, 70), "Measured in ManiSkill 3 with real Muse Spark", font=bold[36], fill=WHITE, anchor="mm")
        y = 135
        for what, ours, others, extra in RESULTS:
            d.text((280, y + 4), ours, font=bold[44], fill=GOOD, anchor="ra")
            d.text((320, y), what, font=reg[26], fill=WHITE)
            d.text((320, y + 40), f"{others}   ·   {extra}", font=reg[22], fill=GREY)
            d.text((320 + d.textlength(f"{others}   ·   ", font=reg[22]), y + 40), extra, font=reg[22], fill=ACCENT)
            y += 100
        d.text((W // 2, H - 50), "ours = Muse Spark + SAM 3.1 + Imagination", font=reg[22], fill=GREY, anchor="mm")
    frames += card(results, 8.0)

    def end(d):
        d.text((W // 2, 300), "ReActSim", font=bold[80], fill=WHITE, anchor="mm")
        d.text((W // 2, 400), "Plan with language. Check with physics. Recover from anything.",
               font=reg[30], fill=ACCENT, anchor="mm")
    frames += card(end, 3.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=1, quality=8)
    print(f"{len(frames) / FPS:.0f} s -> {out}")


if __name__ == "__main__":
    main()
