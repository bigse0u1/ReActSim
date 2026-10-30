# ReActSim

**A robot that imagines before it acts, and recovers when the world fights back.**

ReActSim is a closed-loop robot agent in a physics simulator. You give it an instruction in plain language. **SAM 3.1** finds the objects. **Muse Spark** plans with a fixed set of skills. A Franka Panda carries out the plan in ManiSkill 3, on a table of everyday YCB objects (a banana, a mug, a soup can, a plate). Then we interfere on purpose: knock the banana away, make the gripper drop it, move the target. The agent detects what went wrong from the physics state, explains the cause, and replans until the task is done.

> *From following instructions to understanding failures.*

Built for the **Meta Global AI Developer Hackathon**, *Push the Frontier* track.

---

## What makes it different

| | Typical language-to-robot demo | ReActSim |
|---|---|---|
| Plan | One shot | Closed loop: perceive → plan → act → diagnose → replan |
| Choosing an action | The LLM's first guess | 🧠 **Imagination**: forks the physics, tries each candidate, keeps the best |
| Disturbances | Avoided | 😈 **Chaos Mode**: the audience types perturbations live |
| Success check | The LLM says "done" | The simulator's physical state (pose, contact, velocity) |
| Failure handling | Retry, or nothing | A typed diagnosis (grasp slip, object moved, drop, missed placement) plus a recovery aimed at that cause |

### 🧠 Imagination: real physics lookahead
Before committing to a risky skill (for example choosing between a `top`, `angled` or `side` grasp), the agent snapshots the simulator state (`get_state_dict`). It runs each candidate forward and scores the result with the same diagnoser used at runtime. Then it restores the state and runs only the best option. A grasp is imagined together with the placement that follows it, because a grasp that holds can still make the object tip over when set down. The real run then replays the exact joint commands of the winning rollout (motion planning is random; replaying makes what was imagined what happens, to within 0.3 mm). The planner's proposal stands when it works in physics and is overruled only when it fails. This is actual simulation of the future, not a heuristic feasibility score.

### 😈 Chaos Mode: the world fights back
Anyone can type during a run: *"knock the banana off"*, *"make it drop it"*. Muse Spark turns the text into a bounded physical perturbation. A knock is a real projectile: a YCB baseball is thrown on a ballistic arc at the end of the banana, even while the robot is holding it, and the physics decides whether it twists out of the gripper (at ~6 m/s it did in 4/4 seeds). The robot is not told that anything happened; it has to notice from perception and physics. Scripted chaos makes the benchmarks reproducible.

### 🔍 Failure diagnosis it can explain
Each skill states what should happen. The diagnoser compares that expectation with the observed state and produces a typed `FailureReport` with evidence. Muse Spark receives the report and must address the cause, not blindly repeat the action. When every step "worked" but the success check still fails, the report lists each physics condition with its measured value (e.g. `over_reference (value_cm=20.0)`), and a failure that repeats after the same action is flagged so the planner changes strategy; on the third repeat the agent stops and says why instead of looping.

### 🎥 Show, don't tell: learn from a phone video
Film yourself doing the task with your phone. The robot watches and does it too:

1. **Muse Spark** looks at the first and last frames and names the objects as noun phrases SAM can find (with a fallback word: a green mandarin is "fruit" to SAM, not "orange").
2. **SAM 3.1** tracks the hand and every object through the video (one upload, ~50 frames per prompt at 8 fps).
3. Geometry on the masks finds the object that moved, the frame the hand took it, the frame it came to rest, and what it rests on.
4. **Muse Spark** looks at crops of the grasp moment and names the human's grasp in the robot's vocabulary (`top` / `angled` / `side`).
5. The robot gets the intent plus the human's grasp *as a hint*. A human hand has five fingers, the Panda two, so **Imagination** tries the human's grasp first and adapts when it fails for this body.

Measured on three real clips (home scene, seed 0, Muse Spark + SAM 3.1, all succeeded with no recovery):

| Clip | Human grasp (Muse) | Imagined for the robot | Robot used |
|---|---|---|---|
| mandarin → plate | angled | angled ✓, top ✓, side ✗ (no motion) | angled, as the human did |
| drink can → plate | side | side ✗ (**can tipped over** after placing, 89.5°), top ✓, angled ✓ | angled |
| paper cup → plate | side | side ✗ (lift failed), angled ✓, top ✓ | top |

Each run writes `show_dont_tell.mp4`: the clip with SAM masks on the left, the robot on the right, and what was learned and imagined in between.

### ✋ Same brain, two bodies
`--robot panda` uses a Franka Panda with a two-finger gripper; `--robot hand` uses an xArm7 arm with a five-finger PSYONIC Ability hand (a commercially sold combination). The agent, the prompts, the diagnoser and Imagination are unchanged: only the skills differ, and Imagination finds out what each body can do. The hand grasps by opening all five fingers around the object from above, closing each finger until its pad touches (so the first finger braces instead of pushing), then squeezing.

Pick and place on the plate with the hand (home scene, 3 seeds each, success check incl. standing upright): orange 3/3, can 3/3, cup 3/3, other objects moved by at most 0.2 cm. From the human videos, with Muse Spark: all three succeeded; the human's `side` and `angled` grasps were imagined and rejected for this hand (no feasible path / no angled grasp / the cup slid), and it used its five-finger `top` grasp.

Getting the stock ManiSkill model to this point took several fixes, each found by measuring (details in `src/reactsim/agents/xarm_ability.py`): finger mimic joints blew up under arm motion (q2 now driven directly, 500 Hz physics); the thumb pad kept friction 0.3 because of a typo in the upstream config (`thumnb_L2`); mplib rejected every IK solution because finger hulls overlap; the planner spun joint 1 a full turn into the table; and delta-controlled fingers wound their targets up behind the object, so the hand never reopened.

### 🧺 Several steps: "Put the fruit on the plate and the can in the bowl"
An instruction can ask for several things. Muse Spark turns it into goals (here: banana on plate, orange on plate, can in bowl), grounding words like "the fruit" itself ("Grounded 'the fruit' to both banana and orange"), and orders the work. The task is done only when every goal holds at once; each pick is pre-validated against the goal it serves; a full container gets the next object on its free side; and if a goal that already held comes undone (someone takes the banana back off the plate) the agent notices at that step and fixes it before moving on.

`--scene tidy` (banana, orange, can, a plate and a bowl), three goals, seeds 0-9 (`outputs/bench_tidy`; both systems get the explicit instruction "Put the banana and the orange on the plate, and the can in the bowl", since the rule baseline cannot read "the fruit"):

| Condition | B: rule-based retry | **Muse Spark + Imagination** |
|---|---|---|
| normal | 8/10 | **10/10** |
| knock (ball hits the first object) | 5/10 | **9/10** |
| undo (banana taken off the plate after the second place) | 5/10 | **9/10** |
| total | 18/30 | **28/30** (p=0.005) |
| recoveries per episode | 1.6 / 2.5 / 2.7 | **0.6 / 1.1 / 1.6** |

The gap is much wider than for one object: small failure rates compound over three goals for the retry loop, while pre-validation keeps each step clean. Our only failure in both disturbed conditions is the same layout (seed 5).

### 🧭 Going around: grasps from every side
A grasp other than straight down can come from any side of the object, not only from the robot's: Imagination tries `angled` and `side` from the robot's side, its left, its right and the far side (9 candidates with `top`). Unreachable ones fail in planning almost instantly, so trying nine takes about 3-4 s.

On a cluttered table (`--scene clutter`: the mug and the can stand 12-14 cm from the banana in random directions), seeds 0-19, no disturbance:

| | success | recoveries per episode |
|---|---|---|
| B: rule-based retry (robot's side only) | 15/20 | 1.30 |
| Muse Spark + Imagination, robot's side only | 17/20 | 0.75 |
| **Muse Spark + Imagination, all sides** | **19/20** | **0.35** |

In 5 of the 19 successes the grasp that worked came from the robot's left or right. The one failure is a top grasp that flung the banana 3.6 m (a physics glitch in contact with a neighbour). Twenty seeds do not make the differences significant (p=0.18 vs rule); the drop in recoveries is the clearer effect. The neighbour distance was chosen by measuring: at 14-16 cm nothing is in the way (robot-side grasps solve 7/8 layouts, all sides 7/8); at 12-14 cm 4/8 vs 8/8; at 10-12 cm 3/8 vs 6/8. Results in `outputs/bench_clutter`.

### 🥫 Standing it back up, and nudging on purpose
The baseball usually knocks the soup can over, and "on the plate" means standing. When the arm holds something that must stand but is lying down, the place skill turns the wrist so the can's axis is vertical and sets it down upright; Imagination checks the whole pick-and-place as before. Side grasps also close the fingers horizontally now (the box helper picked the shortest axis, which on a low object is its height, so the gripper stood on its edge against the table; mug side grasps 0/16 → 9/16).

When no grasp works, a candidate that only moves the object is no longer tried blindly: Imagination replays it and tries every grasp again from where it leaves the object. It nudges only if a grasp then works, and grasps right after without a replan; otherwise Muse Spark hears "no grasp works, and nudging does not help either".

Can / knock, seeds 0-19, real Muse Spark, 9 candidates (`outputs/bench_c1/can`):

| | success | recoveries per episode |
|---|---|---|
| before (no stand-up; `outputs/bench_c1/can_before_standup`, 9 seeds) | 0/9 | - |
| B: rule-based retry, with stand-up | 16/20 | 1.8 |
| **Muse Spark + Imagination, with stand-up** | **19/20** | **1.2** |

Standing up is a skill, so it helps the rule baseline too; the remaining gap is not significant (p=0.17). The rule's four failures are "no path" three times in a row or the replan budget; ours is a can thrown 69 cm to the edge of the workspace, out of reach (Muse Spark gave up after two tries and said so). The baseball got stronger after the main benchmark (it now frees a held banana every time), which is why "before" is lower than the 5/20 in the table below.

### 🤝 The LLM proposes, physics disposes
Excerpt from a real run (Muse Spark `muse-spark-1.3`, ManiSkill, seed 0). The banana was knocked out of the gripper by Chaos Mode:

```text
[01] failure  object_moved: banana moved 10.8 cm during the grasp
[01] replan   reobserve → pick(banana) [side] → place_on(plate)
              "Top-down grasp likely slipped off the curved banana surface and pushed it 10.8 cm.
               Re-observing the new pose and retrying with a side grasp for more stable contact."
[02] imagine  pick(banana) [side]    score=-1.00  controller_error   ← no feasible path
[02] imagine  pick(banana) [top]     score=+1.00  none
[02] imagine  pick(banana) [angled]  score=+0.00  grasp_failed
[02] act      pick(banana) [top]
[04] success  banana on plate
```

Muse Spark's reasoning sounded right but its choice would have failed. Imagination tested it in physics and overruled it.

---

## Architecture

```text
 "Put the banana on the plate"            😈 Chaos (scripted / human / adversary)
              │                                         │
              ▼                                         ▼
  ManiSkill 3 RGB-D ──► SAM 3.1 masks ──► depth back-projection ──► WorldState
                                                                      │
                                     ┌────────────────────────────────┘
                                     ▼
                         Muse Spark planner (JSON)
                                     │
                          schema + precondition validator   ✗ → reject & re-ask
                                     │
                         🧠 Imagination (fork → try → restore)
                                     │
                      mplib motion planning skills (pick / place / push)
                                     │
                             ManiSkill physics step
                                     │
                     physics-based diagnosis ──► success ✓
                                     │
                                  failure ──► replan (bounded)
```

**Design boundaries, stated up front:**
- SAM 3.1 masks are not grasp poses. Masks are converted to 3D positions with depth and the camera matrices. The error against ground truth is logged.
- Muse Spark never controls joints. It chooses skills from a whitelist, and an inverse-kinematics motion planner executes them.
- Success is decided by the simulator's physical state, never by an LLM.

## Repository layout

```text
src/reactsim/
  agent.py              ReAct loop
  agents/               xArm7 + Ability hand model (patched for stable simulation and planning)
  types.py              shared dataclasses
  build.py              config → components
  envs/tabletop.py      ReActTabletop-v1: Panda + YCB banana, mug, soup can, plate
  worlds/               World interface; ManiSkill backend; FakeWorld (no-GPU test double)
  perception/           SAM 3.1 / oracle segmenter; mask+depth → 3D
  planning/             Muse Spark client, prompt, JSON schema + validator, baselines
  control/              Panda skills and five-finger hand skills (top / angled / side each)
  imagination/          fork-and-evaluate lookahead (pick + place), exact replay
  imitation/            Show, don't tell: video events + demonstration
  chaos/                scripted + natural-language perturbations
  evaluation/           failure diagnosis, metrics, episode logs
  viz/                  rerun.io live view (optional), side-by-side imitation video
scripts/                run_demo.py, run_benchmark.py, learn_from_video.py, check_env.py, check_api.py
configs/default.yaml
tests/
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Runs without a GPU or API key: FakeWorld + mock planner (checks the pipeline only)
python scripts/run_demo.py --offline
pytest -q

# Real simulation
pip install -e ".[sim,viz]"
python -m mani_skill.utils.download_asset ycb -y     # YCB objects (~85 MB)
python scripts/check_env.py
python scripts/run_demo.py --world maniskill --offline        # sim + mock planner
python scripts/run_demo.py --world maniskill --offline --scene home   # orange, can, cup
cp .env.example .env   # then put MODEL_API_KEY in .env (never in .env.example)
python scripts/check_api.py                                    # one Muse call + SAM masks vs ground truth
python scripts/run_demo.py --world maniskill --chaos-stdin    # sim + Muse Spark + live chaos
python scripts/learn_from_video.py data/demos/can_side.mp4     # phone video -> robot (home scene)
python scripts/make_demo.py --scene clutter --chaos none --seed 2   # demo video: captions + every imagined grasp side by side
python scripts/make_demo.py --scene tidy --chaos undo --instruction "Put the fruit on the plate and the can in the bowl"
python scripts/learn_from_video.py data/demos/can_side.mp4 --robot hand   # ... with the five-finger hand
python scripts/run_demo.py --world maniskill --scene home --robot hand --offline

# Baseline comparison on identical seeds and perturbations
python scripts/run_benchmark.py --world maniskill --seeds 20
```

## Evaluation

| Condition | Description |
|---|---|
| Normal | Fixed objects, no disturbance |
| Position shift | Target moved before or during execution |
| Grasp failure | A grasp mode that slips for the object's pose |
| Chaos | Scripted knocks and drops at fixed steps |
| Occlusion / obstacle *(stretch)* | Target hidden or path blocked; needs `push_aside` |

Compared systems: **A** = fixed script, **B** = rule-based retry, **Ours** = Muse Spark + SAM 3.1 + Imagination.
Metrics: `task_success_rate`, `recovery_success_rate`, `mean_recovery_attempts`, LLM latency, invalid LLM outputs.
Every episode is logged to `outputs/*/episodes.jsonl`.

<!-- RESULTS: fill only from real ManiSkill runs. FakeWorld numbers are not results. -->
**Benchmark** (measured before grasps from every side were added, i.e. with Imagination's 3 robot-side candidates; ManiSkill, kitchen scene, Panda, "put the banana on the plate", oracle segmentation, real Muse Spark `muse-spark-1.3`). The disturbance hits right after the first pick.

Main comparison, seeds 0-19, three objects (rule: `outputs/bench_v2`, `outputs/bench_paper`; Muse alone: `outputs/bench_v3`; ours: `outputs/bench_final`):

| Task / disturbance | B: rule-based retry | Muse Spark, no Imagination | **Muse Spark + Imagination** |
|---|---|---|---|
| banana / knock | 17/20 | 15/20 | 17/20 |
| banana / drop | 12/20 | 14/20 | **20/20** |
| can / knock | 4/20 | - | 5/20 |
| can / drop | 19/20 | - | 18/20 |
| orange / knock | 17/20 | - | 19/20 |
| orange / drop | 20/20 | - | 20/20 |

Only banana / drop is a significant difference (Fisher exact, p=0.003 vs rule, p=0.02 vs Muse alone). Imagination matters when the right grasp depends on how the object landed (a curved banana); a can that stays upright or a round orange is handled by simple retries. Against Muse alone, Imagination halves recovery attempts and Muse calls under *drop* (3.05 to 1.05, 4.0 to 2.0). The knock knocks the can over, after which "upright on the plate" cannot be met by anyone.

**End to end from SAM 3.1** (`outputs/bench_sam2`, banana, seeds 0-19): object positions for planning *and* grasp geometry come from SAM 3.1 masks plus depth (`--segmenter sam3 --grasp-from perception`; no simulator ground truth), for every system:

| Disturbance | B: rule-based retry | Muse Spark, no Imagination | **Muse Spark + Imagination** |
|---|---|---|---|
| drop | 19/20 | 18/20 | **20/20** |
| knock | 15/20 | 10/20 | **20/20** |
| total | 34/40 | 28/40 | **40/40** |
| Muse calls / recoveries per episode | - / 2.0 | 3.8 / 2.8 | **2.2 / 1.2** |

40/40 vs 28/40 without Imagination is p=0.0002; vs the rule baseline p=0.026. The grasp box comes from the masked point cloud: the centre from the top surface (seen whole from above; the plain centroid is pulled toward the camera), the long axis from the spread of the points; its centre lands 0.1-0.8 cm from the true one. Before each pick the arm first moves out of the camera's view: right after a slip the hand hovers over the banana and hides most of it (visible area 2,389 to 736 pixels, position error 0.2 to 7 cm).

**Why the rule baseline swings.** On the same banana / drop task its success depends on the setting: 12/20 with ground-truth geometry, 14/20 when it also clears the view, 19/20 with perceived geometry; ours is 20/20 in all of them. The logs show why: after the forced drop the failure reads as "grasp failed", so the rule switches to its next grasp (angled), which keeps pushing the curved banana away, and whether that second choice happens to work depends on the setting. Imagination tries all three first and goes back to the top grasp that works. (Checked and ruled out: the ground-truth box centre is on the banana; from the same post-drop states both boxes grasp 12/12.)

All four systems, seeds 0-9 (`outputs/bench_main`, before the change described below):

| Condition | A: fixed script | B: rule-based retry | Muse Spark, no Imagination | Muse Spark + Imagination |
|---|---|---|---|---|
| normal | 10/10 | 10/10 | 10/10 | 10/10 |
| knock | 3/10 | 10/10 | 9/10 | 9/10 |
| drop | 0/10 | 8/10 | 8/10 | 10/10 |
| mean Muse calls (knock / drop) | - | - | 2.9 / 3.9 | **1.9 / 2.1** |

What this shows, and what it does not:
- A fixed script fails whenever the world changes; every closed-loop system recovers most of the time.
- Against Muse alone, Imagination needs about half the recovery attempts and Muse calls: grasps are tried in physics first, not on the robot.
- With 10 seeds the rule baseline looked best under *knock* (10/10 vs 9/10); with 20 it is 17/20 vs 19/20. Small samples mislead in both directions; the clear gap is *drop* (12/20 vs 20/20).
- **No working grasp is reported, not attempted.** The 10-seed run showed the agent executing grasps that Imagination had already shown would fail, until the replan budget ran out. Now, when every imagined grasp fails and none even moves the object, the agent tells Muse Spark instead of trying (a grasp that moves the object is still tried: a nudge can leave it graspable). In two of the 40 episodes this fired, and both times Muse Spark changed strategy, pushed the can aside, and succeeded; one of them (knock, seed 7) had failed in both earlier runs.

Measured so far (ManiSkill, 4 seeds each, pick and place on the plate; too few seeds to report as rates):

| Object | `top` | `angled` | `side` |
|---|---|---|---|
| banana | 3/4 | 0/4 | 0/4 |
| mug | 1/4 | 1/4 | 0/4 |
| soup can | 4/4 | 4/4 | 3/4 |

SAM 3.1 on the simulator camera (seed 0), IoU against ground-truth segmentation: banana 0.99, can 0.98, plate 0.99, mug 0.99 (prompted as "cup"; "mug" finds nothing when the handle faces away).

## Status

- [x] Agent loop, validator, diagnosis, imagination, and chaos, tested end to end on FakeWorld
- [x] `ReActTabletop-v1` with YCB objects on ManiSkill 3
- [x] mplib skills: pick (`top` / `angled` / `side`) and place
- [x] Muse Spark planning and replanning in the loop (first full run succeeded)
- [x] SAM 3.1 segmentation via the Model API (checked against ground truth)
- [x] SAM 3.1 inside the agent loop (`--segmenter sam3`): looks at the start, after each failure and on `reobserve`; positions from depth are 0.6–4.2 cm from ground truth
- [x] Grasp from perceived positions (`--grasp-from perception`): SAM 3.1 mask + depth -> oriented box, end to end 36/40
- [ ] Mug grasps (the gripper aims at the hollow centre; needs rim or handle grasps)
- [x] *Show, don't tell*: phone video of a human hand → SAM 3.1 video tracking + Muse Spark → robot, adapting the human's grasp through Imagination (3/3 clips)
- [x] Imagination scores pick + place together and the real run replays the imagined joint commands
- [x] Second body: xArm7 + five-finger Ability hand (`--robot hand`), same agent; pick and place 9/9, human videos 3/3
- [x] Hand: `angled` grasp (same five-finger enclose, approach pitched 50 deg): picks 8/9, but placing from a tilted hand mostly fails (Imagination rejects it unless the path from above is blocked)
- [ ] Hand: `side` grasp (plans fail or the object slides), placing after an angled grasp, Chaos ball tuned for round objects (it throws the orange 1.3 m)
- [x] Demo video (`scripts/make_demo.py`): the real run with a caption log, and each Imagination step as its candidates playing side by side with what the physics returned
- [x] Grasps from every side of the object (9 candidates); cluttered scene
- [x] Several-goal tasks (`--scene tidy`, plate and bowl): language grounding, free spots on a container, undone goals noticed at once; 28/30 vs 18/30
- [ ] Parallel Imagination (candidates in worker processes or GPU environments)
- [ ] Benchmark on ManiSkill and the demo video

## What was built during the hackathon

All code in this repository was written during the hackathon (from 2026-10-26). The pick skill adapts ManiSkill's motion-planning example solutions (Apache-2.0). No pre-existing robot-learning code or trained weights are used.

## Acknowledgements

[ManiSkill 3](https://github.com/haosulab/ManiSkill) · [YCB objects](https://www.ycbbenchmarks.com) · [SAPIEN](https://sapien.ucsd.edu) · [mplib](https://github.com/haosulab/MPlib) · Meta SAM 3.1 and Muse Spark via the [Meta Model API](https://dev.meta.ai/docs/overview) · [rerun](https://rerun.io)
