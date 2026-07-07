# Closed-Loop Task-Success Benchmark (V-JEPA 2-AC)

How we set up, run, score, and log the closed-loop manipulation benchmark. This is the paper-style
**task-success** evaluation for the Phase-1 roadmap, using the customized task set
**grasp / reach_with_object / grasp_and_reach / pick_place**. It is inspired by the V-JEPA 2 robot
tasks (arXiv 2506.09985 Table 3), but plain EE-to-point reach was dropped as uninteresting and
replaced by the 2-goal **grasp_and_reach** composition. The strategy behind it is in
[closed_loop_success_plan.md](closed_loop_success_plan.md); this doc is the operational reference.

> Status (clean-slate rebuild in progress). The benchmark is being rebuilt on **fixed, saved task
> bundles** (below) with **two objects (cup, box)**, replacing the earlier random-per-trial runs.
> The prior random-method reports (`results/benchmarks/closed_loop_full_s200`, `..._s400`) and their
> figures were removed in the clean-slate reset and remain recoverable from git history. New results
> will be published here once the fixed-bundle runs complete.

## What we measure

V-JEPA 2-AC plans the **coarse** end-effector motion by CEM-MPC to a goal image; scripted primitives
handle gripper actions and the post-grasp lift used only for the grasp success test. Success is
judged **only** from hidden privileged MuJoCo truth (object
pose, contacts, velocity, tilt) — the model never sees it. Each rollout records a **continuous
error**, and we compute success at **many precision thresholds** from the same run (a precision
curve).

| task | V-JEPA does | scripted | precision error (delta) |
|---|---|---|---|
| **grasp** | plans only the grasp reach; `goal.png` shows the object just grabbed, gripper closed, still on the table | close at the grasp; scripted lift after the V-JEPA phase tests success | grasp-position error; physical gate = grasped + lifted |
| **reach_with_object** | object **starts already grasped** in `start.png`; plans the held object to a far goal location | gripper stays closed | `‖object_final − goal_object‖` |
| **grasp_and_reach** | object starts on the table; plans to `goal_1` (just grabbed), then to `goal` with the held object at the target | close at `goal_1`; gripper stays closed | `‖object_final − goal_object‖` |
| **pick_place** | grasp reach → vicinity → place across 3 goals on the fixed 4/10/4 schedule | close after `goal_1`; open after final goal | `‖object_final − zone_center‖` |
| **place_with_object** | object **starts already grasped/lifted** in `start.png` (the place half of pick_place); plans to `goal_1` (held in the vicinity of the zone) then `goal` (placed) on a 10/4 schedule | open after the final goal | `‖object_final − zone_center‖` |

Every task records a **continuous Euclidean error (delta)** to its goal and is scored at a **swept
sphere radius `x`** (we report the mean delta and the success rate at several `x`, from tight to
loose — the precision curve). No privileged re-centering is used inside the V-JEPA phase, so
grasp/reach_with_object/grasp_and_reach/pick_place numbers reflect V-JEPA's own positioning.

## Objects (paper-faithful, two per task)

Each task is run on **two objects**, on the **same environment** (only the target geom swaps), for a
clean controlled comparison — matching the paper's Cup vs Box distinction (arXiv 2506.09985, Fig. 14):

| object | geometry | grasp | notes |
|---|---|---|---|
| **cup** | **cube cup**: open-top square box with flat walls | one-wall rim grasp — one finger inside, one outside that wall | forgiving, wide capture (paper's Cup object, box-shaped in sim) |
| **box** | single rigid **block** (~5×4×6 cm) | top-down across the narrow width | needs precise finger width (paper's Box, harder) |

Grasp mechanics (verified quote, arXiv 2506.09985 §4.2): the cup is "grasped by placing one finger
inside the object and gripping around the **rim**" (a miss of the rim fails the grasp); the box has
"many feasible grasping configurations, however... requires more precise gripper control" (finger
width). Our cube cup and rigid box reproduce this contrast.

The scene also carries a few **static distractor** items outside the reachable workspace for visual
realism (present in every scenario, never the target), following the paper's cluttered table.

## Fixed, saved task bundles (reproducible + inspectable)

Scenarios are **not randomized per trial**. A scripted expert generates each scenario **once** and
saves it as an inspectable **task bundle** ([../../src/bench/schema.py](../../src/bench/schema.py)
`TaskBundle`); the benchmark **loads** these bundles, so every config (e.g. sample count, `W*`
calibration, fine-tuned vs vanilla predictor) is scored on the **identical** scenarios.

```
tasks/
  grasp/             cup/  grasp_cup_00/ ... _49/             box/  grasp_box_00/ ... _49/
  reach_with_object/ cup/  ...                                box/  ...
  grasp_and_reach/   cup/  ...                                box/  ...
  pick_place/        cup/  ...                                box/  ...
```

Each `..._NN/` folder holds:

```
meta.json         task, object, difficulty, camera, success_spec, seed
start.png         t0 observation (planner input; for reach_with_object the object starts in the gripper)
goal.png          final goal image (grasp = just grabbed; reach_with_object/grasp_and_reach = held-object target; pick_place = placed)
goal_1.png        grasp_and_reach and pick_place sub-goal 1 (object just grabbed, not lifted)
goal_2.png        pick_place sub-goal 2 (object held in the vicinity of the zone)
arrays.npz        qpos0, object_pose, goal_object, target/zone pose, ee states, camera
contact_sheet.png start | sub-goals | goal strip for human inspection
```

Counts: **4 tasks × 2 objects × 50 scenarios = 400 bundles** (100 trials/task: 50 cup + 50 box). The
50 within each (task, object) differ in object/target placement; each is frozen so the whole suite is
deterministic and re-runnable. Every scenario is validated at generation time — the scripted expert
must complete it — so each saved bundle has a well-defined, measurable hidden success/failure.

A tiny tracked sample set lives under [`../../examples/task_bundles/`](../../examples/task_bundles/):
one scenario for every task/object pair. It is intended for peer inspection and smoke tests; full
benchmark runs should use locally generated bundles under the gitignored `tasks/` directory.

## Paper protocol (verified) and our stage mapping

Goal-image counts are based on the V-JEPA 2 paper protocol (arXiv 2506.09985 §4.2), with one
customization: the plain single-goal reach row is not run because it has no object interaction, and
the 2-goal **grasp_and_reach** task replaces it.

| paper context | # goal images | schedule | our task | our stages | paper V-JEPA 2-AC avg |
|---|---|---|---|---|---|
| Single-goal reaching | **1** | single goal, replan each step | dropped | — | 100% |
| Grasp | **1** | goal = object just grabbed, not lifted | `grasp` | 1 stage + scripted lift gate | Cup 65% / Box 25% |
| Reach with object | **1** | single goal (object starts in hand) | `reach_with_object` | 1 stage | Cup 75% / Box 75% |
| Custom grasp + reach-with-object composition | **2** (`goal_1`, `goal`) | just-grabbed goal, then held-object target | `grasp_and_reach` | 2 stages | n/a |
| Pick-and-Place | **3** (2 sub-goals + final) | **4 / 10 / 4** time-steps | `pick_place` | 3 stages, fixed 4/10/4 | Cup 80% / Box 65% |

Goal images for pick_place: (1) the object just grabbed at its location (not lifted), (2) the object
held in the *vicinity* of the zone, (3) the object *placed* in the zone. Sub-goals switch on a fixed
step budget (4→10→4), not on reaching — so `pick_place` stages use `fixed_steps` (no distance
early-stop).

Honest fidelity notes: (a) the paper's robot uses action clip **maxnorm = L1-ball radius 0.075**
(~13 cm/step) and averages over **10 trials**; we run 50 trials/object. (b) The paper controls the
gripper via the CEM `close_gripper` schedule; we script close/open at stage transitions and the
post-grasp lift used only to score `grasp`, consistent with our V-JEPA-does-spatial /
scripted-does-gripper decomposition.

## How a test runs, and every hyperparameter

The benchmark is a set of nested loops:
**BENCHMARK > TASK > OBJECT > BUNDLE (trial) > TASK stages > MPC time-step (× the 4/10/4 counts) >
CEM (10 iters × N samples × T=2 rollout)**.

| hyperparameter | value | what it does |
|---|---|---|
| **trials** | 50 per (task, object) | fixed saved scenarios per object; more = tighter success-rate estimate |
| **objects** | cup, box | the two target objects; same scene, swapped geom |
| **sub-goal schedule** | grasp: 1 just-grabbed goal; reach_with_object: 1 held-object goal; grasp_and_reach: `goal_1` just grabbed then `goal` held-object target; pick_place: `goal_1`/`goal_2`/`goal` on **4/10/4** | goal-image sequence and, where fixed, number of MPC time-steps spent driving toward each goal image |
| **MPC time-step** | (the counts above) | one closed-loop cycle: observe image+EE state → plan an action with CEM → execute only the 1st action (receding horizon) → re-observe |
| **rollout / T** | 2 | planning horizon — how many future frames the world model predicts per candidate action-trajectory (T=2 = 2 steps ahead). Paper text sometimes says 1; released code = 2 |
| **samples** | 200 / 400 / 800 | # of candidate action-trajectories CEM draws per iteration; more = better search, more VRAM/time |
| **cem_steps** | 10 | # of CEM refinement iterations per MPC step (sample → score → keep topk → re-sample) |
| **topk** | 10 | # of best candidates kept each CEM iteration to update the sampling mean/std |
| **maxnorm** | 0.05 (paper-text 0.075) | per-axis action clip in metres (max EE move/axis/step); also the initial sampling std |
| **momentum_mean / std** | 0.15 / 0.15 | how much the CEM distribution carries over between iterations (smoothing) |
| **pos_tol** | 0.015 | early-stop eligible stages when the EE is within 1.5 cm (pick_place stages run fixed steps, no early-stop) |
| **chunk** | 400 | predictor sub-batch size over the sample dimension — caps peak VRAM (mathematically identical) |
| **dtype** | bf16 | model forward precision |
| **objective** | mean-L1 in layer-norm'd latent | CEM scores each candidate by the L1 distance between its predicted final latent and the goal latent |
| **gripper** | frozen axis | V-JEPA plans only the arm; the gripper (close/lift/open) is scripted at stage transitions |
| **model** | ViT-g encoder (1.01B) + AC predictor (305M) | frozen V-JEPA 2; image 256×256 → 256 tokens/frame |

## Success criteria (hidden state)

Each task reports the continuous Euclidean **delta** to its goal and is scored at a **swept sphere
radius `x`** (`success@x = delta < x` AND the task's physical gates). We report the mean delta and
success@x across `x` from tight to loose, per (task, object):

- **grasp**: V-JEPA plans only to the just-grabbed goal image (closed gripper on the object, still on
  the table). A scripted lift then tests the grasp. `success@x = delta < x` AND `grasped` AND
  `lifted` (object Δz > 4 cm) AND `upright` (<30°) AND `stable` (<5 cm/s). x ∈ {6, 3, 2} cm.
- **reach_with_object**: object starts already grasped in the bundle. delta =
  `‖object_final − goal_object‖`. `success@x = delta < x` AND `held` (never dropped) AND `upright`
  (<30°). x ∈ {10, 6, 3, 1.5} cm.
- **grasp_and_reach**: object starts on the table; V-JEPA first reaches the just-grabbed `goal_1`, then
  moves the held object to the final target. delta = `‖object_final − goal_object‖`. `success@x =
  delta < x` AND `held` (never dropped after the grasp) AND `upright` (<30°). x ∈ {10, 6, 3, 1.5} cm.
- **pick_place**: V-JEPA does grasp → vicinity → place on the fixed 4/10/4 schedule. delta =
  `‖object_final − zone_center‖`. `success@x = delta < x` AND `grasped` (was actually picked up) AND
  `released` AND `upright` (<25°) AND `stable`. Here `released` uses `object_placed` — gripper open
  AND the object resting at table height AND settled — rather than the strict "not touching" test, so
  a correctly-placed rim cup whose inner finger still grazes the wall is not spuriously failed
  (success = where the object landed + the arm let go, per the paper's intent). x ∈ {10, 6, 3, 1.5} cm.

The single source of truth for these radii is `src/bench/thresholds.py::THRESHOLDS`; the generator
imports it, so every bundle's `meta.json` advertises exactly the radii the benchmark scores against.
The final per-trial `success`/`failure` are error-aware: a trial only counts as success when the
object is within the loosest sphere AND every gate holds, and a gates-pass-but-off-target trial is
labelled `off_goal` (`outside_zone` for pick_place) rather than a blank success.

Sweeping `x` low→high yields the precision curve; the mean delta and success@x together show how tight
a tolerance V-JEPA can meet. Failure types are recorded categorically (grasp: missed / not_lifted /
tipped / unstable / off_goal; reach_with_object and grasp_and_reach: dropped / tipped / off_goal;
pick_place: grasp_failed / not_released / tipped / unstable / outside_zone).

## Environment and data

- **Embodiment**: Franka Panda + Robotiq 2F-85 in MuJoCo (`FrankaDroidEnv`), matching V-JEPA 2-AC's
  DROID training embodiment (paper authenticity). The physical target is a UR7e (Stage 2).
- **Observation**: 256×256 RGB from the validated `PLANNING_CAMERA` (az45_el45 exocentric free
  camera), ImageNet-normalized — the exact vendored `make_transforms` path. Plus the 7-D EE state and
  a goal image.
- **Objects / target**: cup (cube cup/open-top square box with one-wall rim grasp) or box (rigid
  block), plus a place-zone marker (5 cm radius) and static distractor clutter.
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

## Bundle loader (how a run executes)

`--bundles <dir>` switches the runner from randomizing per trial to LOADING the fixed scenarios under
`<dir>/<task>/<object>/`. For each bundle a trial runs as:

1. **Restore the exact recorded start** — `env.set_state(qpos0)` loads the saved joint/object state and
   camera; the place zone is restored from the saved `zone` (mocap). `start_grasped` tasks
   (reach_with_object) load with the gripper closed so it holds the object.
2. **Plan to the saved goal images** — each sub-goal is the saved `goal_1/goal_2/goal.png`. CEM-MPC
   replans every step from `(current frame + goal image)`; the gripper axis is frozen (V-JEPA plans
   only the arm).
3. **Auto-switch sub-goals** — like the released V-JEPA 2-AC pick-and-place schedule, sub-goals switch
   on a fixed step budget (pick_place **4 / 10 / 4**); single-goal stages early-stop when the EE
   reaches the target (`pos_tol`).
4. **Script only the gripper** at transitions (close after the grasp goal; open after the place goal)
   and, for `grasp`, a scripted lift that tests the grasp.
5. **Score from hidden privileged state** — Euclidean delta within the swept sphere `x` plus the
   task's physical gates. For **pick_place** success only requires the object to land in the zone and
   be released (the arm's distance from the object is ignored).

The run is deterministic (fixed bundles + seeded CEM); every config (samples, `W*`, fine-tuned vs
vanilla predictor) is scored on the identical scenarios.

## How to run

```
# 1) generate the fixed bundles (scripted expert; CPU/GL, no world model)
python scripts/generate_task_bundles.py --tasks grasp reach_with_object grasp_and_reach pick_place --objects cup box --trials 50

# 2) run the benchmark on the saved bundles (loads tasks/..., deterministic; reports per task x object)
python scripts/run_closed_loop_benchmark.py --bundles tasks --tasks grasp reach_with_object grasp_and_reach --objects cup box --tag full
python scripts/run_closed_loop_benchmark.py --bundles tasks --tasks pick_place --objects cup box --tag full

# defaults follow Meta's released config: samples 200, cem_steps 10, T=2, topk 10, maxnorm 0.05,
# momentum 0.15/0.15. Sample ablation: add --samples 400 (then 800, which uses the chunked predictor).
```

## Demo: ground truth vs V-JEPA

`--demo <task>` builds a side-by-side GIF: the **optimal scripted expert (GROUND TRUTH)** and
**V-JEPA (ours)** driving to the *same* random scenario under the same per-step action clip, played in
sync with a live distance readout. It shows how V-JEPA's planned path compares to the ideal. NOTE:
`--demo` is **legacy only** (the random-scenario reach / grasp_lift / place / pick_place tasks); it
does not load the fixed bundles and the fixed-bundle tasks are not available through it.

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
