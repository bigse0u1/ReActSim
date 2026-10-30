"""Demo video: the real run with a running caption log, and every Imagination step shown
as its candidates playing side by side with what the physics returned.

    rec = DemoRecorder(world)      # use as the agent's dashboard
    agent.run(...)
    rec.render("demo.mp4")

The recorder is a Dashboard: it still prints the log, and also notes at which frame of
the real run each event happened, so the video can pause there.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import numpy as np

from reactsim.viz.dashboard import Dashboard

W, H, FPS = 1280, 720, 20
SLOWMO_SUBSTEPS = 40   # physics sub-steps (100 Hz) rendered one by one during chaos
ZOOM = 0.72   # crop the render to this fraction around the table so the gripper is visible
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, PANEL, WHITE, GREY = (18, 20, 24), (30, 33, 40), (235, 235, 235), (150, 155, 165)
COLOURS = {"plan": (120, 170, 255), "replan": (120, 170, 255), "act": WHITE, "see": GREY,
           "chaos": (255, 120, 90), "failure": (255, 95, 95), "success": (110, 220, 130),
           "give_up": (255, 200, 90), "imagine": (255, 175, 70), "planner": (255, 200, 90)}
LABELS = {"plan": "PLAN", "replan": "REPLAN", "act": "ACT", "see": "SEE", "chaos": "CHAOS",
          "failure": "FAILURE", "success": "SUCCESS", "give_up": "GAVE UP", "imagine": "IMAGINE",
          "planner": "PLANNER"}
FAILURE_WORDS = {"none": "works", "object_moved": "pushes it away", "grasp_failed": "slips",
                 "controller_error": "no path", "goal_not_met": "fails on placing", "dropped": "drops it",
                 "target_lost": "loses sight"}


class DemoRecorder(Dashboard):
    def __init__(self, world, title: str = ""):
        super().__init__(enabled=False)
        self.world = world
        self.title = title
        self.real = world.push_capture()            # the real run's frames
        self.events: list[tuple[int, str, object]] = []   # (real frame index, channel, payload)
        perturb = world.perturb

        def perturb_in_slow_motion(event):
            """Chaos is fast (a ball crosses the scene in 0.1 s, two control steps). While it
            happens, render after every physics sub-step (100 Hz) instead of every other control
            step: the physics is unchanged, the video gets ~10x more frames, i.e. slow motion."""
            log, scene = world._actions, world.u.scene
            every, log.capture_every = log.capture_every, 10 ** 9
            step = scene.step
            count = [0]

            def step_and_capture():
                step()
                count[0] += 1
                # every sub-step through the impact, then every 5th while things settle
                if count[0] <= SLOWMO_SUBSTEPS or count[0] % 5 == 0:
                    self.real.append(world.frame())
            scene.step = step_and_capture
            try:
                return perturb(event)
            finally:
                scene.step = step
                log.capture_every = every
        world.perturb = perturb_in_slow_motion

    def text(self, step, channel, msg):
        super().text(step, channel, msg)
        if channel != "imagine":                     # imagination is shown as video, not text
            self.events.append((len(self.real), channel, msg))

    def imagined(self, step, imagined):
        super().imagined(step, imagined)
        if imagined:
            self.events.append((len(self.real), "imagine", imagined))

    # ---- rendering ----------------------------------------------------------------------
    def render(self, out: str | Path, hold_s: float = 1.5) -> Path:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont

        fonts = {s: ImageFont.truetype(FONT, s) for s in (19, 22)}
        bold = {s: ImageFont.truetype(FONT_BOLD, s) for s in (20, 22, 26, 30)}
        self.world.pop_capture()
        real = self.real or [np.zeros((H, H, 3), np.uint8)]
        log: list[tuple[str, str]] = []
        frames = []

        def fit(im, w, h, zoom=ZOOM):
            ih, iw = im.shape[:2]
            ch, cw = int(ih * zoom), int(iw * zoom)
            y0, x0 = int((ih - ch) * 0.55), (iw - cw) // 2   # the table sits slightly below centre
            img = Image.fromarray(np.ascontiguousarray(im[y0:y0 + ch, x0:x0 + cw]))
            img.thumbnail((w, h))
            canvas = Image.new("RGB", (w, h), BG)
            canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
            return canvas

        def log_panel(draw, x0, y0, w, h):
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=PANEL)
            if self.title:
                draw.text((x0 + 18, y0 + 14), self.title, font=bold[22], fill=WHITE)
            y = y0 + (56 if self.title else 18)
            lines = []
            for channel, msg in log:
                wrapped = textwrap.wrap(msg, 40)[:4] or [""]
                lines.append((channel, wrapped))
            # newest entries at the bottom; drop old ones until the panel fits
            def height(ls):
                return sum(28 + 25 * len(w) + 10 for _, w in ls)
            while lines and height(lines) > h - (y - y0) - 10:
                lines.pop(0)
            for channel, wrapped in lines:
                draw.text((x0 + 18, y), LABELS.get(channel, channel.upper()), font=bold[20],
                          fill=COLOURS.get(channel, WHITE))
                y += 28
                for ln in wrapped:
                    draw.text((x0 + 30, y), ln, font=fonts[19], fill=WHITE if channel != "see" else GREY)
                    y += 25
                y += 10

        def real_frame(i):
            canvas = Image.new("RGB", (W, H), BG)
            canvas.paste(fit(real[min(i, len(real) - 1)], H, H), (0, 0))
            log_panel(ImageDraw.Draw(canvas), H, 0, W - H, H)
            return np.asarray(canvas)

        def imagination(i, imagined, chosen_idx):
            n = len(imagined)
            clips = [im.frames or [real[min(i, len(real) - 1)]] for im in imagined]
            length = max(len(c) for c in clips)
            cols = n if n <= 3 else min(5, -(-n // 2))      # one row up to 3, then two rows
            rows = -(-n // cols)
            gap, label_h, top = 16, (70 if rows == 1 else 52), 78
            tile = min((W - 40 - (cols - 1) * gap) // cols, (H - top - 60 - rows * label_h) // rows, 520)
            small = rows > 1
            out = []
            for t in range(length + int(hold_s * FPS)):
                canvas = Image.new("RGB", (W, H), BG)
                d = ImageDraw.Draw(canvas)
                d.text((W // 2, 38), f"IMAGINATION  -  {n} grasps tried in a copy of the physics first",
                       font=bold[26], fill=COLOURS["imagine"], anchor="mm")
                for k, (im, clip) in enumerate(zip(imagined, clips)):
                    r, c = divmod(k, cols)
                    in_row = min(cols, n - r * cols)
                    x = (W - in_row * tile - (in_row - 1) * gap) // 2 + c * (tile + gap)
                    y = top + r * (tile + label_h)
                    canvas.paste(fit(clip[min(t, len(clip) - 1)], tile, tile), (x, y))
                    d.text((x + tile // 2, y + tile + (14 if small else 22)), grasp_label(im.action, short=small),
                           font=bold[20 if small else 22], fill=WHITE, anchor="mm")
                    if t >= len(clip) - 1:          # outcome appears when this rollout is done
                        ok = not im.failure.is_failure
                        word = FAILURE_WORDS.get(im.failure.type.value, im.failure.type.value)
                        colour = COLOURS["success"] if ok else COLOURS["failure"]
                        d.rectangle([x, y, x + tile, y + tile], outline=colour, width=5)
                        d.text((x + tile // 2, y + tile + (36 if small else 52)), ("OK  " if ok else "X  ") + word,
                               font=bold[20 if small else 22], fill=colour, anchor="mm")
                if t >= length - 1 and chosen_idx is not None:
                    msg = f"-> the robot will do: {grasp_label(imagined[chosen_idx].action)}"
                    d.text((W // 2, H - 30), msg, font=bold[26], fill=COLOURS["success"], anchor="mm")
                elif t >= length - 1:
                    d.text((W // 2, H - 30), "-> nothing works here: tell Muse Spark why instead of trying",
                           font=bold[26], fill=COLOURS["give_up"], anchor="mm")
                out.append(np.asarray(canvas))
            return out

        events = sorted(self.events, key=lambda e: e[0])
        ev = 0
        for i in range(len(real)):
            while ev < len(events) and events[ev][0] <= i:
                _, channel, payload = events[ev]
                if channel == "imagine":
                    chosen = _chosen(payload)
                    frames += [real_frame(i)] * int(0.5 * FPS)
                    frames += imagination(i, payload, chosen)
                else:
                    log.append((channel, readable(channel, str(payload))))
                    if channel in ("failure", "success", "chaos", "give_up", "plan", "replan"):
                        frames += [real_frame(i)] * int(1.0 * FPS)   # let the viewer read it
                ev += 1
            frames.append(real_frame(i))
        while ev < len(events):
            _, channel, payload = events[ev]
            if channel != "imagine":
                log.append((channel, readable(channel, str(payload))))
            ev += 1
        frames += [real_frame(len(real) - 1)] * int(2.5 * FPS)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out, frames, fps=FPS, macro_block_size=1, quality=8)
        return out


def _chosen(imagined) -> int | None:
    """Index Imagination.choose picks (the proposal if it works, else the best score);
    None when every candidate fails and the agent reports instead of acting."""
    from reactsim.types import FailureType

    if not imagined[0].failure.is_failure and imagined[0].result.ok:
        return 0
    if all(i.failure.is_failure and i.failure.type != FailureType.OBJECT_MOVED for i in imagined):
        return None
    return max(range(len(imagined)),
               key=lambda k: (imagined[k].score, imagined[k].failure.type == FailureType.OBJECT_MOVED))


def readable(channel: str, msg: str) -> str:
    """Turn the agent's log line into something a viewer reads at a glance."""
    def action(a: str) -> str:
        m = re.match(r"(\w+)\(([^)]*)\)(?: \[(\w+)\])?", a.strip())
        if not m:
            return a.strip()
        verb, args, grasp = m.groups()
        verb = verb.replace("_", " ")
        return f"{verb} {args.replace(', ', ' on ')}" + (f" ({grasp})" if grasp else "")

    if channel in ("plan", "replan") and msg.startswith("["):
        acts, _, why = msg.partition("] — ")
        steps = [action(a.strip(" '")) for a in acts.strip("[]").split("', '")]
        return " -> ".join(steps) + (f". {why}" if why else "")
    if channel == "act":
        a, _, result = msg.partition(" -> ")
        return f"{action(a)}: {result}"
    if channel == "see":
        names = re.findall(r"(\w+) \(", msg) or re.findall(r"(\w+)", msg.split("|")[0])
        lost = msg.split("not found:")[1].strip() if "not found:" in msg else ""
        return "sees " + ", ".join(names) + (f"; cannot see {lost}" if lost else "")
    if channel == "chaos":
        m = re.search(r"(knock|drop|teleport)\s*(\w*)", msg)
        if m:
            kind, obj = m.groups()
            return {"knock": f"a baseball is thrown at the {obj or 'object'}",
                    "drop": "the gripper is forced open",
                    "teleport": f"someone takes the {obj or 'object'} away"}[kind]
    if channel == "failure":
        kind, _, detail = msg.partition(": ")
        return f"{kind.replace('_', ' ')}: {detail}"
    return msg


# approach_deg turns the approach direction counter-clockwise: at +90 the hand moves toward +y,
# so it comes from the -y side, which is the robot's right (the robot faces +x).
SIDES = {90: "from the robot's right", -90: "from the robot's left", 180: "from the far side"}


SIDES_SHORT = {90: "right", -90: "left", 180: "far side"}


def grasp_label(action, short: bool = False) -> str:
    """'angled from the robot's left', or 'angled (left)' where space is tight."""
    deg = round(action.approach_deg or 0)
    if action.grasp == "top" or deg == 0:
        return action.grasp
    if short:
        return f"{action.grasp} ({SIDES_SHORT.get(deg, f'{deg:+d} deg')})"
    return f"{action.grasp} {SIDES.get(deg, f'from {deg:+d} deg')}"
