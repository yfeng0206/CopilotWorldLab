# Plan

Roadmap and action items for the pilot. The near-term focus is Stage 1: a software pilot
in MuJoCo that exercises the world-model coarse placement and the confidence-gate
measurement, before any physical hardware.

## Stages

- **Stage 1 (current).** Software pilot in simulation: the MuJoCo scene, world-model
  coarse placement (V-JEPA 2-AC, Option 1), and the confidence-gate measurement. Stage 1
  tests whether the gate signal is meaningful *in this simulator*; whether that confidence
  transfers to real hardware remains for Stage 2. Because the end-effector is a kinematic
  mocap body, Stage 1 validates the latent scoring / planning interface, not real
  robot-arm dynamics.
- **Stage 2.** The same orchestrator on the physical UR7e for a single vial-placement
  task: real coarse-placement error, end-to-end seat success, and a setup-cost comparison
  against classical calibration on real holders.
- **Stage 3 (grand plan).** Navigation between stations, real instruments, and more
  chemistry and labware.

## Immediate next steps (next session)

Ordered to fail fast on the cheapest layer first, before spending slow GPU-hours on full
MPC (at ~16 s/action, large-N CEM trials are expensive on one 3090):

1. **Encoder sanity.** Wire `VJEPA2ACWorldModel` to `checkpoints/vjepa2-ac-vitg.pt` via
   `torch.hub`; encode a single rendered 256x256 frame and confirm latent shape
   (16x16x1408). First real inference.
2. **One-step interface calibration.** Generate known sim transitions
   `(x_k, s_k, a_gt, x_{k+1})`; score candidate deltas through the predictor and confirm
   `a_gt` is ranked near the latent-energy minimum. Sweep world- vs body-frame, per-axis
   sign, and translation/rotation scale until the ground-truth action wins.
3. **Camera / cadence ablation.** Repeat (2) with `scene_cam` (third-person, DROID-like)
   vs `wrist_cam`, and with the action interval aligned to ~4 fps (~0.25 s) rather than a
   single sim step. Lock in the interface before planning.
4. **Trivial CEM reach.** Only now implement `plan_action` (following
   `facebookresearch/vjepa2/notebooks/utils/mpc_utils.py`); validate the energy decreases
   toward a captured goal image on a one-step reach. Measure latency on the 3090.
5. **Franka integration.** Swap the mocap end-effector for a Franka from MuJoCo Menagerie
   behind the same 7-D interface (pose->IK shim), adding real arm/gripper dynamics before
   any confidence-gate conclusions.
6. **Trial harness + gate data.** Sample a goal pose, perturb the start pose synthetically
   (standing in for docking error), plan, and log terminal energy, the confound baselines
   (below), and the success label.

## Stage-1 build checklist

- [x] Reproducible local environment (venv + CUDA Torch + MuJoCo), verified on the 3090.
- [x] Minimal MuJoCo scene and `MujocoPilotEnv` (render, 7-DoF EE, goal capture).
- [x] V-JEPA 2-AC interface scaffold and download-only checkpoint fetch.
- [x] Test suite (geometry, env kinematics, render) passing.
- [ ] Encoder-only inference on a rendered frame.
- [ ] CEM planning to a goal image; latency measurement.
- [ ] Trial harness + confidence-gate data collection.
- [ ] Gate evaluation (ROC AUC vs baseline and vs pixel-error convergence).
- [ ] Franka (MuJoCo Menagerie) behind the same 7-D interface.

## Evaluation (the two primary measurements)

1. **The confidence gate.** Whether the world model's confidence predicts a failed
   handoff, reported as ROC AUC against a simple baseline and against the visual servo's
   pixel-error convergence. If it does not beat these, the combine-and-gate approach does
   not hold. A negative result is itself an informative finding.
2. **Setup cost.** Demonstrations (and wall-clock time) to reach a target seating rate on
   a new holder, versus the time to calibrate the same holder by the classical procedure,
   over at least two holders. The learned approach is worthwhile only if the per-holder
   learned cost, once one-time pretraining and simulation are amortised over many
   stations, falls below the classical calibration cost.

Secondary: coarse placement error against the capture range, and end-to-end seating
success with bounded retry. All reported with trial counts and confidence intervals.

### Confounds, baselines, and anti-selection design

The terminal energy is also what CEM minimises, so a naive ROC AUC can overstate the
gate's quality (low-energy attempts are preferentially selected and handed off). To avoid
this:

- **Evaluate on a fixed policy over all attempts** -- successes, failures, retries, and
  no-handoff cases included -- not only on executed/handed-off trials. Calibrate the
  threshold on held-out trials and report AUC plus the false-accept rate at the chosen
  operating point.
- **Define the "failed handoff" label explicitly and consistently** (do not mix
  definitions across trials). Candidate proxy until the visual servo exists: the coarse
  pose is outside the servo's capture range (or, later, the simulated servo fails to
  converge / pixel error stays above threshold).
- **Compare the energy against confound baselines**, since low goal-energy may just mean
  "visually close to goal", not "model competence": encoder-only latent distance
  `||z_k - z_g||`; a pixel / pretrained-visual-feature distance; privileged sim pose
  error; CEM improvement margin and top-k variance; prediction self-consistency
  `||P(a;s,z_k) - encode(x_{k+1})||`; random/permuted-action energy; and the visual
  servo's pre-handoff pixel error. The gate is only interesting if it beats these.

## Open decision

The sub-millimetre fine seat: classical visual servo vs a fully unified latent (see
[`DESIGN.md`](DESIGN.md) Section 6). Recommended path is the hybrid. The pilot is built to
inform this empirically; it is not resolved yet.

## Risks

- Simulation-to-real transfer of the learned primitives is the main risk (no public
  dataset contains chemistry labware; data is generated in simulation plus a few
  teleoperated demonstrations).
- V-JEPA 2-AC's stated limitations -- camera/coordinate-frame sensitivity, long-horizon
  drift, 16 s/action latency, monocular RGB -- are expected to dominate and are mirrored
  in the evaluation.
