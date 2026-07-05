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

No privileged re-centering is used inside the V-JEPA phase, so grasp/place numbers reflect
V-JEPA's own positioning. (The place task scripts the *initial* grasp to isolate the placement
skill.)

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
maxnorm **0.05 m/axis** · momentum 0.15 · bf16 · objective = mean-L1 in layer-norm'd latent with
the gripper axis frozen · receding-horizon replan. (Released `world_model_wrapper.py` defaults;
the paper text quotes a larger population ~800 and may report horizon 1 — we ablate T=1 vs T=2 and
samples 200/400/800 later. See [../architecture.md](../architecture.md#7-planner-config-verified-from-released-code).)

## How to run

```
# smoke (wiring / a few trials)
python scripts/run_closed_loop_benchmark.py --tasks reach grasp_lift place --trials 5 --tag smoke

# full benchmark (50 trials/task, precision curves)
python scripts/run_closed_loop_benchmark.py \
    --tasks reach grasp_lift place --trials 50 \
    --samples 200 --cem-steps 10 --rollout 2 --maxnorm 0.05
```

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

### Smoke, 5 trials/task (config above, seed 0)

| task | success (loosest τ) | mean V-JEPA final error | read |
|---|---|---|---|
| **Reach** | **5/5** | 3–7 cm | pure V-JEPA closed-loop is reliable |
| **Grasp-Lift** | **3/5** | 5.5–7.7 cm | V-JEPA positions the gripper well enough to grasp ~60%; misses are `missed` |
| **Place** | **0/5** | **15–18 cm** | vanilla V-JEPA can't precisely place the held cube — plateaus ~15 cm vs the 6 cm zone (all `outside_zone`) |

Honest reading: reach is easy; grasp is decent; **place exposes the precision gap** — the held-
object horizontal placement plateaus well outside the zone. This is the vanilla baseline the
improvements (W* frame calibration, predictor fine-tuning, and the POV/cross-view method,
Phases 2–4) must beat, measured on this exact protocol.

Representative rollouts (`results/benchmarks/closed_loop_smoke/`): `reach_success.gif`,
`grasp_success.gif`, `grasp_missed.gif`, `place_outside_zone.gif` (+ contact sheets);
`trials_5trial.csv` has all 15 trials. The full 50-trial precision-curve run is prepared but not
yet run.

## Reproduce

Config, seeds, and per-step data are logged under `logs/closed_loop_runs/<run_id>/` for exact
diagnosis. The runner is `scripts/run_closed_loop_benchmark.py`; success functions are pure and
unit-tested (`src/bench/success.py`, `tests/test_success.py`); the grasp physics has a regression
test (`tests/test_bench_env.py`).
