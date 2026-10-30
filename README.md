# ReActSim

**A robot that imagines before it acts, and recovers when the world fights back.**

ReActSim is a closed-loop robot agent in a physics simulator. You give it an instruction in plain language. **SAM 3.1** finds the objects. **Muse Spark** plans with a fixed set of skills. A Franka Panda carries out the plan in ManiSkill 3, on a table of everyday YCB objects (a banana, a mug, a soup can, a plate). Then we interfere on purpose: knock the banana away, make the gripper drop it, move the target. The agent detects what went wrong from the physics state, explains the cause, and replans until the task is done.

> *From following instructions to understanding failures.*

Built for the **Meta Global AI Developer Hackathon**, *Push the Frontier* track.

**Results** (real ManiSkill runs, real Muse Spark, two-sided Fisher exact tests):

| | Muse Spark + Imagination | Rule-based retry | Muse Spark alone |
|---|---|---|---|
| 3 objects × 2 disturbances, 20 seeds each | **116/120** | 105/120 (p=0.015) | – |
| From SAM 3.1 perception only (banana) | **40/40** | 34/40 (p=0.026) | 28/40 (p=0.0002) |
| Three goals, two containers | **28/30** | 18/30 (p=0.005) | – |

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
Before a pick, the agent snapshots the simulator state (`get_state_dict`). It tries 9 grasp candidates: `top`, plus `angled` and `side` from the robot's side, its left, its right and the far side of the object. Each candidate runs together with the placement that follows it, because a grasp that holds can still make the object tip over when set down. Each result is scored with the same diagnoser used at runtime, and the state is restored. The planner's proposal stands when it works in physics and is overruled only when it fails. The real run then replays the exact joint commands of the winning rollout: motion planning is random, and replaying makes what was imagined what happens, to within 0.3 mm. Unreachable candidates fail in planning almost instantly, so trying nine takes about 3-4 s.

When no grasp works, the agent does not act. A candidate that only moves the object is imagined one step further, and the agent nudges only if a grasp then works. Otherwise Muse Spark hears which grasps were tried and why each failed, and changes strategy (pushes a neighbour aside, or gives up and says why).

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

### 😈 Chaos Mode: the world fights back
Anyone can type during a run, in English or Korean. Muse Spark turns the sentence into a bounded physical perturbation. A knock is a real projectile: a YCB baseball is thrown on a ballistic arc at the object, even while the robot is holding it, and the physics decides whether it comes loose. The robot is not told that anything happened; it has to notice from perception and physics. Scripted chaos makes the benchmarks reproducible.

Muse Spark sees where every movable object is and what the robot holds, so it can resolve "it" or "the cup" and pick a direction. Its answer is checked before anything happens: only movable objects (never the plate), a move stays on the workspace and within 25 cm, "drop" needs something in the hand, and a sentence that moves nothing is ignored. Real Muse Spark readings:

| Spectator | Muse Spark |
|---|---|
| "knock it out of the robot's hand" | knock banana: *Baseball knocks banana out of gripper* |
| "slide the can far to the left" | move can, dy = +0.25 m |
| "바나나를 로봇 손에서 쳐내" | knock banana: *Baseball knocks banana from gripper* |
| "컵을 오른쪽으로 옮겨" ("move the cup to the right") | move mug, dy = -0.15 m |

### 🔍 Failure diagnosis it can explain
Each skill states what should happen. The diagnoser compares that expectation with the observed state and produces a typed `FailureReport` with evidence. Muse Spark receives the report and must address the cause, not blindly repeat the action. When every step "worked" but the success check still fails, the report lists each physics condition with its measured value (e.g. `over_reference (value_cm=20.0)`). A failure that repeats after the same action is flagged so the planner changes strategy; on the third repeat the agent stops and says why instead of looping.

### 🧺 Several steps: "Put the fruit on the plate and the can in the bowl"
An instruction can ask for several things. Muse Spark turns it into goals (here: banana on plate, orange on plate, can in bowl), grounds words like "the fruit" itself, and orders the work. The task is done only when every goal holds at once; each pick is pre-validated against the goal it serves; and if a goal that already held comes undone (someone takes the banana back off the plate), the agent notices at that step and fixes it.

`--scene tidy`, three goals, seeds 0-9. Both systems get the explicit instruction "Put the banana and the orange on the plate, and the can in the bowl", since the rule baseline cannot read "the fruit":

| Condition | Rule-based retry | **Muse Spark + Imagination** |
|---|---|---|
| no disturbance | 8/10 | **10/10** |
| knock (ball hits the first object) | 5/10 | **9/10** |
| undo (banana taken off the plate after the second place) | 5/10 | **9/10** |
| total | 18/30 | **28/30** (p=0.005) |
| recoveries per episode | 1.6 / 2.5 / 2.7 | **0.6 / 1.1 / 1.6** |

Small failure rates compound over three goals for the retry loop, while pre-validation keeps each step clean.

### 🧭 Going around, and standing it back up
On a cluttered table (`--scene clutter`: the mug and the can stand 12-14 cm from the banana in random directions, 20 seeds), grasps from every side succeed 19/20, against 17/20 for robot-side grasps only and 15/20 for the rule baseline (recoveries per episode 0.35 / 0.75 / 1.30; not significant with 20 seeds). In 5 of the 19 successes the grasp that worked came from the robot's left or right.

The baseball usually knocks the soup can over, and "on the plate" means standing. When the arm holds something that must stand but is lying down, the place skill turns the wrist so it goes down upright. Can / knock went from 0/9 to 19/20 (rule baseline with the same skill: 16/20).

### 🎥 Show, don't tell: learn from a phone video
Film yourself doing the task with your phone. The robot watches and does it too:

1. **Muse Spark** looks at the first and last frames and names the objects as noun phrases SAM can find.
2. **SAM 3.1** tracks the hand and every object through the video.
3. Geometry on the masks finds the object that moved, when the hand took it, and what it was put on.
4. **Muse Spark** looks at the grasp moment and names the human's grasp in the robot's vocabulary (`top` / `angled` / `side`).
5. The robot gets the task plus the human's grasp *as a hint*. **Imagination** tries the human's grasp and adapts when it fails for this body.

Three real clips (mandarin, drink can, paper cup → plate) all succeeded with no recovery. For the drink can, the human's side grasp was imagined and rejected (the can tipped over after placing, 89.5°), and the robot used an angled grasp instead. Each run writes `show_dont_tell.mp4`: the clip with SAM masks on the left, the robot on the right, and what was learned and imagined in between.

### ✋ Same brain, two bodies
`--robot panda` uses a Franka Panda with a two-finger gripper; `--robot hand` uses an xArm7 arm with a five-finger PSYONIC Ability hand. The agent, the prompts, the diagnoser and Imagination are unchanged: only the skills differ, and Imagination finds out what each body can do. The hand closes each finger until its pad touches the object, then squeezes. Pick and place with the hand: orange, can and cup 9/9; the phone videos 3/3.

---

## Architecture

```text
 "Put the banana on the plate"            😈 Chaos (scripted / typed by a spectator)
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

**Design boundaries:**
- SAM 3.1 masks are not grasp poses. Masks are turned into 3D positions and oriented boxes with depth and the camera matrices (centre 0.1-0.8 cm from ground truth; mask IoU 0.98-0.99).
- Muse Spark never controls joints. It chooses skills from a whitelist, and a motion planner executes them.
- Success is decided by the simulator's physical state, never by an LLM.

## Evaluation

Compared systems: **rule-based retry** (on failure, look again and cycle through grasp modes), **Muse Spark alone** (no Imagination), and **ours** (Muse Spark + Imagination). Same seeds, same disturbance right after the first pick: *knock* (the baseball hits the held object) or *drop* (the gripper is forced open). ManiSkill, Panda, real Muse Spark `muse-spark-1.3`, seeds 0-19. Every episode is logged to `outputs/*/episodes.jsonl`. All p-values are two-sided Fisher exact tests.

| Task / disturbance | Rule-based retry | Muse Spark alone | **Muse Spark + Imagination** |
|---|---|---|---|
| banana / knock | 18/20 | 18/20 | **19/20** |
| banana / drop | 13/20 | 17/20 | **20/20** |
| can / knock | 16/20 | - | **19/20** |
| can / drop | 20/20 | - | 20/20 |
| orange / knock | 18/20 | - | **19/20** |
| orange / drop | 20/20 | - | 19/20 |
| total | 105/120 | | **116/120** (p=0.015) |
| recoveries per episode, banana (knock / drop) | 1.2 / 2.5 | 2.3 / 2.9 | **0.8 / 1.0** |
| Muse calls per episode, banana (knock / drop) | - | 3.3 / 3.9 | **1.9 / 2.0** |

Banana / drop is the clearest cell (20/20 vs 13/20, p=0.008). After the forced drop the failure reads as "grasp failed", so the rule switches to its next grasp (angled), which keeps pushing the curved banana away. Imagination tries all of them first and goes back to the top grasp that works. A can that lands upright or a round orange is handled by simple retries, so there the gap is small. Against Muse alone the success gap is not significant (p=0.23), but Imagination halves recovery attempts and Muse calls: grasps are tried in physics first, not on the robot.

**End to end from SAM 3.1** (banana, seeds 0-19): object positions for planning *and* grasp geometry come from SAM 3.1 masks plus depth (`--segmenter sam3 --grasp-from perception`; no simulator ground truth), for every system. Measured with the earlier 3 robot-side candidates:

| Disturbance | Rule-based retry | Muse Spark alone | **Muse Spark + Imagination** |
|---|---|---|---|
| drop | 19/20 | 18/20 | **20/20** |
| knock | 15/20 | 10/20 | **20/20** |
| total | 34/40 | 28/40 | **40/40** |
| Muse calls / recoveries per episode | - / 2.0 | 3.8 / 2.8 | **2.2 / 1.2** |

Before each pick the arm moves out of the camera's view: right after a slip the hand hovers over the banana and hides most of it (position error grows from 0.2 to 7 cm).

## Limitations

- Simulation only. Imagination runs in a perfect copy of the simulator; on a real robot the scene would have to be rebuilt from perception, and how that error affects pre-validation is untested.
- Twenty seeds per cell: small differences (19/20 vs 18/20) are not significant.
- The Panda cannot grasp the mug from above (it aims at the hollow centre), and the hand's side grasp rarely works.
- Imagination tries candidates one after another (3-4 s for nine); they could run in parallel.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Without a GPU or API key: FakeWorld + mock planner (checks the pipeline only)
python scripts/run_demo.py --offline
pytest -q

# Real simulation
pip install -e ".[sim,viz]"
python -m mani_skill.utils.download_asset ycb -y     # YCB objects (~85 MB)
python scripts/check_env.py
python scripts/run_demo.py --world maniskill --offline        # sim + mock planner
cp .env.example .env   # then put MODEL_API_KEY in .env (never in .env.example)
python scripts/check_api.py                                    # one Muse call + SAM masks vs ground truth

python scripts/run_demo.py --world maniskill --chaos-stdin    # Muse Spark + type disturbances live
python scripts/make_demo.py --chaos knock --seed 1            # demo video: captions + every imagined grasp
python scripts/make_demo.py --chaos none --seed 1 --say "knock the banana out of the robot's hand"
python scripts/make_demo.py --scene tidy --chaos undo --instruction "Put the fruit on the plate and the can in the bowl"
python scripts/learn_from_video.py data/demos/can_side.mp4 [--robot hand]   # phone video -> robot
python scripts/make_final_video.py                             # stitch the demo videos into one

python scripts/run_benchmark.py --world maniskill --seeds 20  # baselines on identical seeds and disturbances
```

## Repository layout

```text
src/reactsim/
  agent.py              ReAct loop
  agents/               xArm7 + Ability hand model (patched for stable simulation and planning)
  types.py              shared dataclasses
  build.py              config → components
  envs/tabletop.py      ReActTabletop-v1: scenes kitchen, home, clutter, tidy
  worlds/               World interface; ManiSkill backend; FakeWorld (no-GPU test double)
  perception/           SAM 3.1 / oracle segmenter; mask + depth → 3D
  planning/             Muse Spark client, prompt, JSON schema + validator, baselines
  control/              Panda skills and five-finger hand skills
  imagination/          fork-and-evaluate lookahead (pick + place), exact replay
  imitation/            Show, don't tell: video events + demonstration
  chaos/                scripted + natural-language perturbations
  evaluation/           failure diagnosis, metrics, episode logs
  viz/                  demo and imitation videos, rerun.io live view (optional)
scripts/                run_demo, run_benchmark, make_demo, make_final_video, learn_from_video, check_env, check_api
configs/default.yaml
tests/
```

## What was built during the hackathon

All code in this repository was written during the hackathon (from 2026-10-26). The pick skill adapts ManiSkill's motion-planning example solutions (Apache-2.0). No pre-existing robot-learning code or trained weights are used.

## Acknowledgements

[ManiSkill 3](https://github.com/haosulab/ManiSkill) · [YCB objects](https://www.ycbbenchmarks.com) · [SAPIEN](https://sapien.ucsd.edu) · [mplib](https://github.com/haosulab/MPlib) · Meta SAM 3.1 and Muse Spark via the [Meta Model API](https://dev.meta.ai/docs/overview) · [rerun](https://rerun.io)
