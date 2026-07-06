# Closed-Loop Task-Success Benchmark (V-JEPA 2-AC)

How we set up, run, score, and log the closed-loop manipulation benchmark. This is the paper-style
**task-success** evaluation (Reach / Grasp / Pick-Place), the Phase-1 deliverable of the roadmap
([../DESIGN.md](../DESIGN.md#0-project-roadmap-phases)). The strategy behind it is in
[closed_loop_success_plan.md](closed_loop_success_plan.md); this doc is the operational reference.

> Status (clean-slate rebuild in progress). The benchmark is being rebuilt on **fixed, saved task
> bundles** (below) with **two objects (cup, box)**, replacing the earlier random-per-trial runs.
> The prior random-method reports (`results/benchmarks/closed_loop_full_s200`, `..._s400`) and their
> figures were removed in the clean-slate reset and remain recoverable from git history. New results
> will be published here once the fixed-bundle runs complete.

## What we measure

V-JEPA 2-AC plans the **coarse** end-effector motion by CEM-MPC to a goal image; scripted primitives
handle only the gripper. Success is judged **only** from hidden privileged MuJoCo truth (object
pose, contacts, velocity, tilt) — the model never sees it. Each rollout records a **continuous
error**, and we compute success at **many precision thresholds** from the same run (a precision
curve).

| task | V-JEPA does | scripted | precision error (delta) |
|---|---|---|---|
| **Reach** | plans EE to the goal image (object present in scene as context) | — | `‖EE_final − target‖` |
| **Grasp** | plans the grasp reach (goal = arm at the object) | close **+ lift** | grasp-position error `‖object_xy − EE_xy‖`; success gate = grasped + lifted |
| **Reach-with-object** | object **starts grasped + lifted** in the bundle state; plans the **held** object to the goal (no scripted lift) | gripper stays closed | `‖object_final − goal_object‖` |
| **Pick-Place** | grasp reach → transport → place across 3 sub-goals (4/10/4); **no scripted lift/transport** | close after grasp; open after place | `‖object_final − final_goal‖` |

Every task records a **continuous Euclidean error (delta)** to its goal and is scored at a **swept
sphere radius `x`** (we report the mean delta and the success rate at several `x`, from tight to
loose — the precision curve). No privileged re-centering is used inside the V-JEPA phase, so
grasp/reach-with-object/pick-place numbers reflect V-JEPA's own positioning.

## Objects (paper-faithful, two per task)

Each task is run on **two objects**, on the **same environment** (only the target geom swaps), for a
clean controlled comparison — matching the paper's Cup vs Box distinction (arXiv 2506.09985, Fig. 14):

| object | geometry | grasp | notes |
|---|---|---|---|
| **cup** | procedural thin-wall **open cylinder** (rim-graspable) | rim grasp — fingers span opposite walls | forgiving, wide capture (paper's Cup) |
| **box** | single rigid **block** (~5×4×6 cm) | top-down across the narrow width | needs precise finger width (paper's Box, harder) |

Grasp mechanics (verified quote, arXiv 2506.09985 §4.2): the cup is "grasped by placing one finger
inside the object and gripping around the **rim**" (a miss of the rim fails the grasp); the box has
"many feasible grasping configurations, however... requires more precise gripper control" (finger
width). Our procedural rim-cup and rigid box reproduce this contrast.

The scene also carries a few **static distractor** items outside the reachable workspace for visual
realism (present in every scenario, never the target), following the paper's cluttered table.

## Fixed, saved task bundles (reproducible + inspectable)

Scenarios are **not randomized per trial**. A scripted expert generates each scenario **once** and
saves it as an inspectable **task bundle** ([../../src/bench/schema.py](../../src/bench/schema.py)
`TaskBundle`); the benchmark **loads** these bundles, so every config (e.g. sample count, `W*`
calibration, fine-tuned vs vanilla predictor) is scored on the **identical** scenarios.

```
tasks/
  reach/             cup/  reach_cup_00/ ... _49/            box/  reach_box_00/ ... _49/
  grasp/             cup/  ...                                box/  ...
  reach_with_object/ cup/  ...                                box/  ...
  pick_place/        cup/  ...                                box/  ...
```

Each `..._NN/` folder holds:

```
meta.json         task, object, difficulty, camera, success_spec, seed
start.png         t0 observation (planner input; for reach_with_object the object starts in the gripper)
goal.png          final goal image (planner target)
goal_1.png        pick_place sub-goal 1 (object grasped)
goal_2.png        pick_place sub-goal 2 (object in vicinity of the zone)
arrays.npz        qpos0, object_pose, goal_object_pose, target/zone pose, ee states, camera
model.xml         exact patched MJCF for this scenario
contact_sheet.png start | sub-goals | goal strip for human inspection
```

Counts: **4 tasks × 2 objects × 50 scenarios = 400 bundles** (100 trials/task: 50 cup + 50 box). The
50 within each (task, object) differ in object/target placement; each is frozen so the whole suite is
deterministic and re-runnable. Every scenario is validated at generation time — the scripted expert
must complete it — so each saved bundle has a well-defined, measurable hidden success/failure.

## Paper protocol (verified) and our stage mapping

Goal-image counts per task are taken directly from the V-JEPA 2 paper (arXiv 2506.09985 §4.2,
verified against the PDF, not a summary):

| paper task | # goal images | schedule | our task | our stages | paper V-JEPA 2-AC avg |
|---|---|---|---|---|---|
| Single-goal reaching | **1** | single goal, replan each step | `reach` | 1 stage | 100% |
| Grasp | **1** | single goal | `grasp` | 1 stage | Cup 65% / Box 25% |
| Reach with object | **1** | single goal (object starts in hand) | `reach_with_object` | 1 stage | Cup 75% / Box 75% |
| Pick-and-Place | **3** (2 sub-goals + final) | **4 / 10 / 4** time-steps | `pick_place` | 3 stages, fixed 4/10/4 | Cup 80% / Box 65% |

Paper goal images for pick-and-place: (1) the object being grasped, (2) the object in the *vicinity*
of the goal, (3) the object *at* the goal. Sub-goals switch on a fixed step budget (4→10→4), not on
reaching — so `pick_place` stages use `fixed_steps` (no distance early-stop).

Honest fidelity notes: (a) the paper's robot uses action clip **maxnorm = L1-ball radius 0.075**
(~13 cm/step) and averages over **10 trials**; we run 50 trials/object. (b) The paper controls the
gripper via the CEM `close_gripper` schedule; we script the close/open at stage transitions,
consistent with our V-JEPA-does-spatial / scripted-does-gripper decomposition.

## How a test runs, and every hyperparameter

The benchmark is a set of nested loops:
**BENCHMARK > TASK > OBJECT > BUNDLE (trial) > TASK stages > MPC time-step (× the 4/10/4 counts) >
CEM (10 iters × N samples × T=2 rollout)**.

| hyperparameter | value | what it does |
|---|---|---|
| **trials** | 50 per (task, object) | fixed saved scenarios per object; more = tighter success-rate estimate |
| **objects** | cup, box | the two target objects; same scene, swapped geom |
| **sub-goal schedule** | reach 5; grasp 6; reach_with_object 8; pick_place **4/10/4** | number of MPC time-steps spent driving toward each goal image; 4/10/4 = the paper's grasp→vicinity→place budget (3 goals) |
| **MPC time-step** | (the counts above) | one closed-loop cycle: observe image+EE state → plan an action with CEM → execute only the 1st action (receding horizon) → re-observe |
| **rollout / T** | 2 | planning horizon — how many future frames the world model predicts per candidate action-trajectory (T=2 = 2 steps ahead). Paper text sometimes says 1; released code = 2 |
| **samples** | 200 / 400 / 800 | # of candidate action-trajectories CEM draws per iteration; more = better search, more VRAM/time |
| **cem_steps** | 10 | # of CEM refinement iterations per MPC step (sample → score → keep topk → re-sample) |
| **topk** | 10 | # of best candidates kept each CEM iteration to update the sampling mean/std |
| **maxnorm** | 0.05 (paper-text 0.075) | per-axis action clip in metres (max EE move/axis/step); also the initial sampling std |
| **momentum_mean / std** | 0.15 / 0.15 | how much the CEM distribution carries over between iterations (smoothing) |
| **pos_tol** | 0.015 | early-stop a reach/grasp stage when the EE is within 1.5 cm (pick_place stages run fixed steps, no early-stop) |
| **chunk** | 400 | predictor sub-batch size over the sample dimension — caps peak VRAM (mathematically identical) |
| **dtype** | bf16 | model forward precision |
| **objective** | mean-L1 in layer-norm'd latent | CEM scores each candidate by the L1 distance between its predicted final latent and the goal latent |
| **gripper** | frozen axis | V-JEPA plans only the arm; the gripper (close/lift/open) is scripted at stage transitions |
| **model** | ViT-g encoder (1.01B) + AC predictor (305M) | frozen V-JEPA 2; image 256×256 → 256 tokens/frame |

## Success criteria (hidden state)

Each task reports the continuous Euclidean **delta** to its goal and is scored at a **swept sphere
radius `x`** (`success@x = delta < x` AND the task's physical gates). We report the mean delta and
success@x across `x` from tight to loose, per (task, object):

- **Reach**: delta = `‖EE_final − target‖`. `success@x = delta < x`. x ∈ {5, 3, 1.5} cm.
- **Grasp**: V-JEPA reaches the grasp pose, then a **scripted close + lift**. Success gate = `grasped`
  (held, gripper–object contact) AND `lifted` (object Δz > 4 cm) AND `upright` (<30°) AND `stable`
  (<5 cm/s); delta = grasp-position error, reported over x ∈ {6, 3, 2} cm.
- **Reach-with-object**: object starts grasped + lifted in the bundle. delta =
  `‖object_final − goal_object‖`. `success@x = delta < x` AND `held` (never dropped) AND `upright`
  (<30°). x ∈ {6, 3, 1.5} cm.
- **Pick-Place**: V-JEPA does grasp → transport → place (no scripted lift). delta =
  `‖object_final − final_goal‖`. `success@x = delta < x` AND `grasped` at hand-off AND `released`
  (gripper open, not touching) AND `upright` (<25°) AND `stable`. x ∈ {10, 6, 3, 1.5} cm.

Sweeping `x` low→high yields the precision curve; the mean delta and success@x together show how tight
a tolerance V-JEPA can meet. Failure types are recorded categorically (grasp: missed / pushed /
slipped / tipped / dropped; reach_with_object: dropped / tipped / off_goal; pick_place: grasp_failed /
outside_zone / tipped / unstable / still_attached).

## Environment and data

- **Embodiment**: Franka Panda + Robotiq 2F-85 in MuJoCo (`FrankaDroidEnv`), matching V-JEPA 2-AC's
  DROID training embodiment (paper authenticity). The physical target is a UR7e (Stage 2).
- **Observation**: 256×256 RGB from the validated `PLANNING_CAMERA` (az45_el45 exocentric free
  camera), ImageNet-normalized — the exact vendored `make_transforms` path. Plus the 7-D EE state and
  a goal image.
- **Objects / target**: cup (procedural rim-graspable open cylinder) or box (rigid block), plus a
  place-zone marker (5 cm radius) and static distractor clutter.
- **Data source**: states are **generated in simulation** by a scripted expert and **saved as fixed
  bundles** — no external dataset is downloaded for this benchmark, and there is **no robomimic
  dependency** (robomimic replay is a separate reference tool, not part of this closed loop).
- **Trials**: 50 per (task, object), deterministic (loaded from `tasks/…`).

## CEM planning config (verified from Meta's released code)

model V-JEPA 2-AC ViT-g · samples **200** · cem_steps **10** · topk **10** · rollout **T=2** ·
maxnorm **0.05 m/axis** · momentum_mean 0.15 · momentum_std **0.15** · pos_tol **0.015 m** · bf16 ·
objective = mean-L1 in layer-norm'd latent with the gripper axis frozen · receding-horizon replan.
(Matches Meta's released `world_model_wrapper.py` robot config. samples=200/T=2 fits the 3090
(~16 GB); the 400/800 sample ablations use the chunked predictor to avoid OOM. See
[../architecture.md](../architecture.md#7-planner-config-verified-from-released-code).)

## How to run

```
# 1) generate the fixed bundles (scripted expert; CPU/GL, no world model)
python scripts/generate_task_bundles.py --tasks reach grasp reach_with_object pick_place --objects cup box --trials 50

# 2) run the benchmark on the saved bundles (loads tasks/..., deterministic)
python scripts/run_closed_loop_benchmark.py --bundles tasks --tasks reach grasp reach_with_object --objects cup box --tag full
python scripts/run_closed_loop_benchmark.py --bundles tasks --tasks pick_place --objects cup box --tag full

# 3) side-by-side ground-truth vs V-JEPA demo GIF
python scripts/run_closed_loop_benchmark.py --demo reach
```

## Demo: ground truth vs V-JEPA

`--demo <task>` builds a side-by-side GIF: the **optimal scripted expert (GROUND TRUTH)** and
**V-JEPA (ours)** driving to the *same* saved scenario under the same per-step action clip, played in
sync with a live distance readout. It shows how V-JEPA's planned path compares to the ideal.

## Logging and outputs

Every run writes two places:

- **Full run log (gitignored, for diagnosis)** — `logs/closed_loop_runs/<run_id>/`:
  - `run_config.json` — the complete inference setup: model, checkpoint SHA256, git commit, device,
    dtype, all CEM params, thresholds, gate spec, env params, normalization, seeds, bundle ids.
  - `steps.csv` — every step of every trial (phase, energy, planned + realized action, EE/object/
    target xyz, error, obj_dz, tilt, speed, held, released, CEM time, success, failure).
  - `trials.csv` — per-trial bundle id, error, loosest-threshold pass/fail, failure, final latent
    energy, V-JEPA vs total steps, mean CEM time, and per-threshold success flags (JSON).
  - `viz/` — GIF + phase-keyed contact sheet + markdown frame table for the ~3 best / 3 median /
    3 worst trials per (task, object).
- **Committed report** — `results/benchmarks/closed_loop_<tag>/<run_id>/`: `summary.md`,
  `summary.csv`, `<task>_<object>_summary.png` (error histogram, precision curve, failure bars,
  error-vs-latent-energy scatter), and the selected GIFs/contact sheets.

Visual overlays on every frame: red = object center, blue = EE/gripper, green = target-zone circle
(3-D world points projected into the planning camera), plus a stats panel (task, object, trial,
step, phase, energy, action, realized, EE/object/target xyz, error, obj_dz, tilt, speed, held,
released, success/failure).

## Results

Pending the fixed-bundle rebuild. The precision curves (per task × object, at samples 200/400/800)
will be published here once the runs complete. The earlier random-per-trial numbers were removed in
the clean-slate reset and are recoverable from git history if a historical comparison is needed.

## References

- Assran et al. *V-JEPA 2.* arXiv:2506.09985 (2025). Closed-loop task success, §4.2 + App. B/Fig. 14.
- Success functions: [../../src/bench/success.py](../../src/bench/success.py); thresholds:
  [../../src/bench/thresholds.py](../../src/bench/thresholds.py); bundle schema:
  [../../src/bench/schema.py](../../src/bench/schema.py).
