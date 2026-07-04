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

Done so far: encoder + AC predictor load; CEM latency measured (~32 s at 800 samples);
interface calibration via the **camera ablation** (view-relative frame; `PLANNING_CAMERA` =
az45_el45 is the `FrankaDroidEnv` default); **benchmark 1** (transition scoring) vanilla baseline
(rank 1.00 vs null 0.30, AUROC 0.953 DROID); and **Phase 1 closed-loop CEM** to a goal image
(`docs/experiments/cem_closed_loop.md`) -- reach succeeds (goal image in 3 steps), multi-goal
chaining works, and the ~3 cm precision floor is diagnosed as a model/interface limit (tracking
error only 9 mm), motivating W* + fine-tuning. The plan is benchmark-driven
(`docs/experiments/benchmark_plan.md`). Remaining:

1. **W* calibration + re-run Phase 1.** Fit/freeze the App. B.4 horizontal rotation for the
   planning camera and re-run the CEM chain; expect the lateral sub-goal to converge -- the first
   improvement delta on our own closed-loop benchmark.
2. **ManiSkill (benchmark 2).** Separate venv (SAPIEN); adapter render -> V-JEPA latent -> the
   Phase-1 CEM loop -> step -> official success on PickCube / StackCube / PegInsertionSide.
3. **Harden benchmark 1 on a DROID batch.** Score many real DROID trajectories for a statistical
   rank_frac/AUROC (needs the DROID dataset in the npz format).
4. **Predictor fine-tuning + re-benchmark.** Fine-tune the predictor (frozen encoder) on small
   task data; re-run benchmarks 1-2 + Phase 1 and report improvement as metric deltas.
5. **Confidence-gate data.** Perturb the start pose, plan, log terminal energy + confound
   baselines + success label (the project's central gate measurement).

## Stage-1 build checklist

- [x] Reproducible local environment (venv + CUDA Torch + MuJoCo), verified on the 3090.
- [x] Minimal MuJoCo scene and `MujocoPilotEnv` (render, 7-DoF EE, goal capture).
- [x] V-JEPA 2-AC interface scaffold and download-only checkpoint fetch.
- [x] Test suite (geometry, env kinematics, render) passing.
- [x] Franka Panda (MuJoCo Menagerie) loaded, rendered, actuated, and timed (smoke test).
- [x] Franka + Robotiq 2F-85 composed (mjSpec), exocentric camera, EE-space control via
      differential IK (`FrankaDroidEnv`); scripted reach test passes 5/5.
- [x] `apply_action` dynamically stepped (IK -> ctrl -> mj_step) with a measured gripper
      opening and action bounds (commit abeaad6; the substrate for grasp / world-model wiring).
- [x] Encoder + AC predictor load from the local checkpoint; encoder-only + CEM inference
      run in `scripts/vjepa2_ac_infer_test.py`.
- [x] CEM latency measured (V-JEPA 2-AC, bf16 on the 3090: 800 samples = 32 s, chunked).
- [ ] CEM planning to a rendered goal image, in the env loop (interface calibration first).
- [ ] Trial harness + confidence-gate data collection.
- [ ] Gate evaluation (ROC AUC vs baseline and vs pixel-error convergence).

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
