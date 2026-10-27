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
Before committing to a risky skill (for example choosing between a `top`, `angled` or `side` grasp), the agent snapshots the simulator state (`get_state_dict`). It runs each candidate forward and scores the result with the same diagnoser used at runtime. Then it restores the state and runs only the best option. This is actual simulation of the future, not a heuristic feasibility score. The dashboard shows the imagined outcomes side by side.

### 😈 Chaos Mode: the world fights back
Anyone can type during a run: *"knock the banana off"*, *"make it drop it"*. Muse Spark turns the text into a bounded physical perturbation. The robot is not told that anything happened; it has to notice from perception and physics. Scripted chaos makes the benchmarks reproducible.

### 🔍 Failure diagnosis it can explain
Each skill states what should happen. The diagnoser compares that expectation with the observed state and produces a typed `FailureReport` with evidence. Muse Spark receives the report and must address the cause, not blindly repeat the action.

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
  types.py              shared dataclasses
  build.py              config → components
  envs/tabletop.py      ReActTabletop-v1: Panda + YCB banana, mug, soup can, plate
  worlds/               World interface; ManiSkill backend; FakeWorld (no-GPU test double)
  perception/           SAM 3.1 / oracle segmenter; mask+depth → 3D
  planning/             Muse Spark client, prompt, JSON schema + validator, baselines
  control/skills.py     pick (top / angled / side) / place_on / push_aside on mplib
  imagination/          fork-and-evaluate lookahead
  chaos/                scripted + natural-language perturbations
  evaluation/           failure diagnosis, metrics, episode logs
  viz/dashboard.py      rerun.io live view (optional)
scripts/                run_demo.py, run_benchmark.py, check_env.py, check_api.py
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
cp .env.example .env   # then put MODEL_API_KEY in .env (never in .env.example)
python scripts/check_api.py                                    # one Muse call + SAM masks vs ground truth
python scripts/run_demo.py --world maniskill --chaos-stdin    # sim + Muse Spark + live chaos

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
*Results: to be filled in from ManiSkill runs.*

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
- [ ] Grasp from perceived positions (planning uses SAM + depth; the pick skill still targets the simulator's object pose)
- [ ] Mug grasps (the gripper aims at the hollow centre; needs rim or handle grasps)
- [ ] *Show, don't tell*: learn the task from a phone video of a human hand (SAM 3.1 video tracking + Muse Spark)
- [ ] rerun dashboard with the imagined-outcomes view
- [ ] Benchmark on ManiSkill and the demo video

## What was built during the hackathon

All code in this repository was written during the hackathon (from 2026-10-26). The pick skill adapts ManiSkill's motion-planning example solutions (Apache-2.0). No pre-existing robot-learning code or trained weights are used.

## Acknowledgements

[ManiSkill 3](https://github.com/haosulab/ManiSkill) · [YCB objects](https://www.ycbbenchmarks.com) · [SAPIEN](https://sapien.ucsd.edu) · [mplib](https://github.com/haosulab/MPlib) · Meta SAM 3.1 and Muse Spark via the [Meta Model API](https://dev.meta.ai/docs/overview) · [rerun](https://rerun.io)
