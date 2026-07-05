# Closed-Loop Task-Success Benchmark (V-JEPA 2-AC)

How we set up, run, score, and log the closed-loop manipulation benchmark, and the current
results. This is the paper-style **task-success** evaluation (Reach / Grasp-Lift / Place), the
Phase-1 deliverable of the roadmap ([../DESIGN.md](../DESIGN.md#0-project-roadmap-phases)). The
strategy behind it is in [closed_loop_success_plan.md](closed_loop_success_plan.md); this doc is
the operational reference and results page.

## What we measure

V-JEPA 2-AC plans the **coarse** end-effector motion by CEM-MPC to a goal image; scripted
primitives handle only the gripper. Success is judged **only** from hidden privileged MuJoCo
truth (object pose, contacts, velocity, tilt) — the model never sees it. Instead of one arbitrary
pass/fail cutoff, each rollout records a **continuous error**, and we compute success at **many
precision thresholds** from the same run (a precision curve).

| task | V-JEPA does | scripted | precision error |
|---|---|---|---|
| **Reach** | full closed-loop to a goal image | — | `‖EE_final − target‖` |
| **Grasp-Lift** | reaches the grasp pose (goal = arm at the cube) | close + lift only | `‖object_xy − EE_xy‖` before close |
| **Place** | drives the held cube over the zone | scripted grasp to start; lower-straight-down + open | `‖object_xy_final − zone_xy‖` |
| **Pick-Place** | the WHOLE composite: grasp reach → transport → place, across 3 sub-goals (4/10/4) | close after grasp; lower + open after place | `‖object_xy_final − zone_xy‖` |

No privileged re-centering is used inside the V-JEPA phase, so grasp/place numbers reflect
V-JEPA's own positioning. (The isolated *place* task scripts the *initial* grasp to isolate the
placement skill; the *pick_place* task does not — V-JEPA does the grasp too.)

## Paper protocol (verified) and our stage mapping

The number of goal images per task is taken directly from the V-JEPA 2 paper (arXiv 2506.09985
§4.2, verified against the PDF, not a summary):

| paper task | # goal images | schedule | our task | our stages |
|---|---|---|---|---|
| Single-goal reaching | **1** | single goal, replan each step | `reach` | 1 stage |
| Grasp | **1** | single goal | `grasp_lift` | 1 stage (paper-faithful default) |
| Reach-with-object | **1** | single goal | (not implemented) | — |
| Pick-and-Place | **3** (2 sub-goals + final) | **4 / 10 / 4** time-steps | `pick_place` | 3 stages, fixed 4/10/4 |

Paper goal images for pick-and-place: (1) the object being grasped, (2) the object in the
*vicinity* of the goal, (3) the object *at* the goal. Sub-goals switch on a fixed step budget
(4→10→4), not on reaching — so `pick_place` stages use `fixed_steps` (no distance early-stop).

Two honest notes on fidelity: (a) the paper's robot uses action clip **maxnorm = L1-ball radius
0.075** (~13 cm/step) and averages over **10 trials**; we default to maxnorm 0.05 (the released-code
value) and run more trials — 0.075 is a documented ablation, not our default. (b) The paper controls
the gripper via the CEM `close_gripper` schedule; we script the close/open at stage transitions,
consistent with our V-JEPA-does-spatial / scripted-does-gripper decomposition. Our **multistage
grasp** (pregrasp→grasp) is *our* addition, not the paper's single-goal grasp, so it is reported
only as a labeled ablation (`--protocol multistage`).

## Success criteria (hidden state)

A trial succeeds at precision threshold `τ` iff `error < τ` **AND** all physical gates hold:

- **Reach**: `error < τ`. Thresholds τ ∈ {5, 3, 1.5} cm.
- **Grasp-Lift**: `error < τ` AND `lifted` (object Δz > 4 cm) AND `held` (gripper–object contact)
  AND `upright` (tilt < 30°) AND `stable` (speed < 5 cm/s). Thresholds τ ∈ {6, 5, 3, 2} cm.
- **Place**: `error < τ` AND `upright` (tilt < 25°) AND `stable` (speed < 5 cm/s) AND `released`
  (gripper open and not touching the object). Thresholds τ ∈ {10, 6, 3, 1.5} cm.

Failure types are recorded categorically (grasp: missed / pushed / slipped / tipped / dropped;
place: outside_zone / tipped / unstable / still_attached).

## Environment and data

- **Embodiment**: Franka Panda + Robotiq 2F-85 in MuJoCo (`FrankaDroidEnv`), matching V-JEPA
  2-AC's DROID training embodiment (paper authenticity). The physical target is a UR7e (Stage 2).
- **Observation**: 256×256 RGB from the validated `PLANNING_CAMERA` (az45_el45 exocentric free
  camera), ImageNet-normalized (mean/std ×255 on 0–255 input) — the exact vendored `make_transforms`
  path. Plus the 7-D EE state and a goal image.
- **Object / target**: a 4 cm graspable free-joint cube (high friction, ~16 g) and a place-zone
  marker (6 cm radius).
- **Data source**: states are **generated in simulation** — no external dataset is downloaded for
  this benchmark. (The DROID download is only for the separate transition-scoring benchmark,
  [transition_scoring.md](transition_scoring.md).) Cube and reach-target positions are
  **randomized per trial** (seeded): cube xy ∈ [0.45, 0.55] × [−0.15, −0.05] m; reach target is a
  seeded offset from home.
- **Trials**: smoke = 1–5 per task (wiring); full = **50 per task**.

## CEM planning config (verified from Meta's released code)

model V-JEPA 2-AC ViT-g · samples **200** · cem_steps **10** · topk **10** · rollout **T=2** ·
maxnorm **0.05 m/axis** · momentum_mean 0.15 · momentum_std **0.15** · pos_tol **0.015 m** · bf16 ·
objective = mean-L1 in layer-norm'd latent with the gripper axis frozen · receding-horizon replan.
(Matches Meta's released `world_model_wrapper.py` robot config; `momentum_std=0.15` and
`pos_tol=0.015` set the loop to keep refining down to the tightest precision threshold. The paper
text quotes a larger population ~800 and may report horizon 1 — we ablate T=1 vs T=2 and samples
200/400/800 later. samples=200/T=2 fits the 3090 (~16 GB); **the 400/800 ablations require porting
the chunked predictor (`vjepa2_ac_infer_test.py`) first to avoid OOM** — do not attempt them until
then. See [../architecture.md](../architecture.md#7-planner-config-verified-from-released-code).)

## Protocols: single-goal vs multistage

The released `cem()` takes a single `goal_frame`, so a paper-like multi-sub-goal schedule is an
**outer loop that swaps the goal image between stages** (`--protocol multistage`), each stage
running single-goal CEM. `--protocol single_goal` uses one goal image per task (baseline).

- **Reach**: one stage in both protocols.
- **Grasp-Lift**: single_goal = one grasp goal; multistage = **pregrasp** (arm hovering above the
  cube) then **grasp** (arm around the cube). Only close+lift are scripted.
- **Place**: single_goal = one held-cube-over-zone goal; multistage = **vicinity** (held cube high
  over the zone) then **final** (held cube lowered onto the zone). The release lower+open is
  scripted. The place sub-goals keep the cube *held* (V-JEPA controls the held EE and cannot drive
  the gripper away without dragging the cube), so a held-low goal is the reachable placement target.

## How to run

```
# smoke comparison (5 trials/task, both protocols)
python scripts/run_closed_loop_benchmark.py --protocol single_goal --tasks reach grasp_lift place --trials 5 --tag single_goal_smoke
python scripts/run_closed_loop_benchmark.py --protocol multistage  --tasks reach grasp_lift place --trials 5 --tag multistage_smoke

# paper-faithful pick-and-place (3 sub-goals, fixed 4/10/4)
python scripts/run_closed_loop_benchmark.py --tasks pick_place --trials 50 --tag full

# reach / grasp full benchmark (grasp single-goal = paper-faithful)
python scripts/run_closed_loop_benchmark.py --tasks reach grasp_lift --trials 50 --tag full

# side-by-side ground-truth vs V-JEPA demo GIF
python scripts/run_closed_loop_benchmark.py --demo reach
```

## Demo: ground truth vs V-JEPA

`--demo reach` builds `results/benchmarks/closed_loop_smoke/demo_reach_compare.gif`: the **optimal
straight-line reach (GROUND TRUTH)** and **V-JEPA (ours)** driving to the *same* seeded target under
the same per-step action clip, played in sync with a live distance readout. It shows how V-JEPA's
planned path compares to the ideal (GT ~1 cm vs V-JEPA ~3 cm on the reference target).

## Logging and outputs

Every run writes two places:

- **Full run log (gitignored, for diagnosis)** — `logs/closed_loop_runs/<run_id>/`:
  - `run_config.json` — the complete inference setup: model, checkpoint SHA256, git commit,
    device, dtype, all CEM params, thresholds, gate spec, env params, normalization, seeds.
  - `steps.csv` — every step of every trial (phase, energy, planned + realized action, EE/object/
    target xyz, error, obj_dz, tilt, speed, held, released, CEM time, success, failure).
  - `trials.csv` — per-trial error, loosest-threshold pass/fail, failure, final latent energy,
    V-JEPA vs total steps, mean CEM time, and per-threshold success flags (JSON).
  - `viz/` — GIF + phase-keyed contact sheet + markdown frame table for the ~3 best / 3 median /
    3 worst trials per task (not all 50, to keep it light).
- **Committed report** — `results/benchmarks/closed_loop_<tag>/<run_id>/`: `summary.md`,
  `summary.csv`, `<task>_summary.png` (error histogram, precision curve, failure bars,
  error-vs-latent-energy scatter), and the selected GIFs/contact sheets.

Visual overlays on every frame: red = object center, blue = EE/gripper, green = target-zone
circle (3-D world points projected into the planning camera), plus a stats panel (task, trial,
step, phase, energy, action, realized, EE/object/target xyz, error, obj_dz, tilt, speed, held,
released, success/failure).

## Results

### Smoke comparison, 5 trials/task, single-goal vs multistage (config above, seed 0)

Each rollout records one continuous error; `success@t` = error < t AND all physical gates. n=5 is a
wiring/comparison check, not a converged rate. Full report + GIFs:
[results/benchmarks/closed_loop_smoke/](../../results/benchmarks/closed_loop_smoke/).

| task | protocol | mean err (cm) | success @ thresholds | physical gates |
|---|---|---|---|---|
| **Reach** | both (identical) | 2.4 | @5cm **100%**, @3cm 80%, @1.5cm 20% | — |
| **Grasp-Lift** | single_goal | 2.24 | @6cm 60%, @3cm 40%, @2cm 20% | held 3/5 |
| **Grasp-Lift** | **multistage** | **1.68** | @6cm **80%**, @3cm **80%**, @2cm **60%** | held **4/5** |
| **Place** | single_goal | 15.6 | @10cm **0%** … @1.5cm 0% | released 5/5 |
| **Place** | multistage | 16.6 | @10cm **0%** … @1.5cm 0% | released 5/5 |

Honest reading:

- **Reach** is a reliable coarse skill.
- **Multistage clearly helps grasp**: the pregrasp → grasp top-down approach doubles @3cm success
  (40% → 80%), lowers error (2.24 → 1.68 cm), and raises held 3/5 → 4/5.
- **Multistage does not help place** (16.6 vs 15.6 cm, both 0%): the vicinity → final descent adds
  error with no meaningful horizontal waypoint — our place is short-horizon, unlike the paper's
  from-scratch pick-and-place.
- **Place stays poor after the goal-image fix — likely a real precision/salience limit, to confirm
  at 50 trials.** Fixing the malformed place goal image (the held cube is now rigidly carried over
  the zone rather than left behind at its grasp site — verified in `goal_image_check.png`) did
  **not** improve place (~15.6 cm, was ~15–18 cm). At n=5 this suggests a placement-precision /
  object-salience limitation rather than the old goal-image artifact; a plausible cause is that
  V-JEPA's latent similarity is dominated by the large arm/gripper pose, not the small cube, so the
  object's position in the goal image has little planning leverage. This is a preliminary reading —
  the 50-trial run is needed before any strong claim. Either way, place is the vanilla baseline the
  improvements (W* frame calibration, predictor fine-tuning, POV/cross-view; Phases 2–4) must beat.

Note on reproducibility: seeds pin the RNG (cube/target placement, CEM samples), but bf16 GPU
forward kernels are non-deterministic, so per-trial errors vary run-to-run; the 50-trial run
reports distributions (mean/median/percentiles), which are stable, rather than exact per-trial
values.

Representative rollouts (`results/benchmarks/closed_loop_smoke/`): `reach_success.gif`,
`grasp_multistage_success.gif`, `grasp_single_missed.gif`, `place_fail.gif` (+ contact sheets);
`comparison_5trial.csv` has all 30 trials. The full 50-trial precision-curve run is prepared but
not yet run.

## Reproduce

Config, seeds, and per-step data are logged under `logs/closed_loop_runs/<run_id>/` for exact
diagnosis. The runner is `scripts/run_closed_loop_benchmark.py`; success functions are pure and
unit-tested (`src/bench/success.py`, `tests/test_success.py`); the grasp physics has a regression
test (`tests/test_bench_env.py`).
