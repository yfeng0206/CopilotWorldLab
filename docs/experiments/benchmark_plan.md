# Benchmark Plan — Honest Evaluation of V-JEPA 2-AC and Our Improvements

## Principle

Do not invent an easy custom dataset and declare victory on it. First test **vanilla**
V-JEPA 2-AC honestly on established tasks, then define *improvement* as measurable deltas on
those **same** benchmarks after our method (better energy calibration and/or predictor
fine-tuning). This avoids the "we made our own easy dataset" problem and keeps the claim
falsifiable.

The strongest claim this project can support:

> We improve V-JEPA 2-AC as a robot world model by making its latent energy better calibrated
> for manipulation success/failure, and/or by fine-tuning the predictor so image-goal planning
> succeeds more reliably on standard manipulation tasks — measured on the same public benchmarks
> the vanilla model is scored on.

## What "improve" means (measurable, not vibes)

| Metric | Meaning |
|---|---|
| Success rate | More tasks completed |
| Sample efficiency | Same success with fewer fine-tuning demos |
| Planning efficiency | Same success with fewer CEM samples / less time |
| Energy calibration | Latent energy better predicts failure / success |
| Generalization | Trained on some tasks/objects, improves held-out tasks/objects |
| Robustness | Better under camera shifts, distractors, object-pose randomization |

Every claim we make must move one of these numbers on an established benchmark.

## Benchmark stack (low custom-data burden, in order)

### 1. Transition prediction / action ranking — IMPLEMENTED and RUN

Closest to V-JEPA, no custom environment. Given `(image_t, state_t, action_t)`, a correct
world model scores the true action lower in latent energy than random negative actions.
[`scripts/benchmark_transition_scoring.py`](../../scripts/benchmark_transition_scoring.py)
reports the within-transition `rank_frac` (fraction of same-magnitude random negatives with
higher energy than the true action; chance 0.5) as the primary metric, plus `top1_acc`,
`gap_z` (effect size), a shuffled-goal **null control**, and a pooled AUROC.

**Vanilla baseline (bf16, RTX 3090, K=32 negatives):**

| transition set | rank_frac | null (shuffled goal) | top1 | gap_z | AUROC (pooled) |
|---|---|---|---|---|---|
| DROID paper example (native, fwd+rev) | **1.00** | 0.30 | 1.00 | +3.06 | 0.953 |

The **null control** is the key honesty check: with the *correct* goal the true action beats
all negatives (rank 1.00), but scored against a *mismatched* (shuffled) goal the same action is
not favored (null 0.30). This is **evidence that the vanilla model is image-goal-conditioned on
the paper example**, not a fixed action prior. It is a smoke-scale baseline (`n=2`, the vendored
example fwd+rev); it becomes a robust established benchmark only when run over many real DROID
trajectories (see Scaling). Actions here are xyz-translation only (rotation/gripper zeroed).

Per-camera on our MuJoCo renders (n=18 each; tracks the camera-placement ablation exactly).
The primary sim result is per-camera because a cross-camera aggregate (rank 0.75, null 0.55)
blends the calibrated and uncalibrated view-relative interfaces:

| camera | rank_frac | null | | camera | rank_frac | null |
|---|---|---|---|---|---|---|
| az45_el45 (planning cam) | **0.958** | 0.670 | | az135_el20 | 0.672 | 0.450 |
| az45_el20 | 0.944 | 0.616 | | az135_el45 | 0.655 | 0.488 |
| az90_el20 | 0.854 | 0.616 | | top_down | 0.615 | 0.533 |
| az90_el45 | 0.849 | 0.589 | | exo_named (built-in) | **0.476** | 0.408 |

Reading: on DROID-native transitions the vanilla model clearly understands the dynamics
(true action beats all 32 negatives, and the null control confirms it uses the goal). Zero-shot
to our simulator, the calibrated planning camera nearly matches DROID while the built-in exo_cam
is at chance — consistent with the view-relative frame finding
([energy_landscape_and_camera_ablation.md](energy_landscape_and_camera_ablation.md)). This is
the **baseline our fine-tuned predictor must beat** on the same metric. It is a one-step scoring
benchmark, not closed-loop planning success (that is the ManiSkill layer below).

Scaling: today's DROID number uses the vendored single example trajectory (n=2). To harden it,
score a batch of real DROID trajectories (dataset download; same script, `--traj`).

### 2. ManiSkill standard tasks — established sim benchmark (next)

Official success labels, no manual dataset. Tasks: PickCube, StackCube, PegInsertionSide,
PlugCharger/insertion. Metrics: zero-shot success rate, success after predictor fine-tuning,
CEM time/samples, and energy-vs-success ROC-AUC. Answers: does V-JEPA 2-AC transfer to standard
simulated manipulation, and does our method raise success / calibration?

Status/compat: `mani-skill` 3.0.1 and `sapien` 3.0.3 install, but **do not run on this Windows
setup** (verified 2026-07-04, separate venv): the end-effector control mode needs Pinocchio
(no Windows wheel -> `PinocchioModel is None`), and even joint control crashes the SAPIEN
native sim with an access violation. Like robosuite (lessons_learned #11), ManiSkill requires
**Linux / WSL2**. So the established-suite closed-loop benchmark is gated on a Linux
environment. Because the V-JEPA scoring/planning code is backend-agnostic (frames + 7-D EE
deltas), only a thin adapter changes: render observation -> V-JEPA latent -> the Phase-1 CEM
loop -> step the env -> official success. Two options: (a) run ManiSkill under WSL2/Linux, or
(b) treat the working MuJoCo `FrankaDroidEnv` as the closed-loop platform and add proper
pick/place tasks with success labels there (needs the graspable-object scene).

### 3. robomimic / LIBERO — only if needed

Established imitation-learning demos; higher integration overhead. Use their existing demos in a
separate env if MuJoCo/robosuite conflict. Deferred unless (2) is insufficient.

### 4. Custom labware env — last (application demo, not the first benchmark)

The Franka + vial + holder scene is an *application demo* of the end goal, not the first
benchmark. Build it after the method is proven on (1)-(2), so we do not spend months building a
dataset before proving the method. This intentionally deprioritizes the earlier "unified scene".

## Minimal first research loop

1. Run vanilla V-JEPA 2-AC on transition scoring. **DONE** — baseline table above.
2. Run vanilla V-JEPA 2-AC on 2-3 ManiSkill tasks (zero-shot success). NEXT (separate venv).
3. Fine-tune only the predictor on small task data (frozen encoder; see
   [vjepa2_ac_architecture.md](../vjepa2_ac_architecture.md)).
4. Re-run the exact same benchmarks (1) and (2).
5. Report improvement as deltas on the metrics table (rank_frac/AUROC, success rate, CEM
   efficiency, energy calibration).

## Where our current work fits (honest boundary)

- The energy-landscape reproduction is a **correctness gate** (the model matches the paper), not
  a benchmark.
- The camera-placement ablation is **interface calibration** (which view/frame to plan in), not
  a benchmark.
- Benchmark (1), transition scoring, is the **first real benchmark with a vanilla baseline**.
- No closed-loop planning success has been measured yet; that is benchmark (2).

## Reproducibility

- Benchmark 1 (DROID example): `python scripts/benchmark_transition_scoring.py`.
- Benchmark 1 (sim, per camera): `python scripts/benchmark_transition_scoring.py --traj "outputs/transitions/*.npz"`
  (render first with `scripts/render_franka_transitions.py --step 0.06 --poses 3`).
- Metrics are seeded (`--seed`, K negatives per transition); per-run CSV in `logs/`.
