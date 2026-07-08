# Plan

Roadmap and action items for the pilot. The near-term focus is Stage 1: a software pilot
in MuJoCo that exercises the world-model coarse placement and the confidence-gate
measurement, before any physical hardware.

## Stages

- **Stage 1 (current).** Software pilot in simulation: the MuJoCo scene, world-model
  coarse placement (V-JEPA 2-AC, Option 1), and the confidence-gate measurement. Stage 1
  tests whether the gate signal is meaningful *in this simulator*; whether that confidence
  transfers to real hardware remains for Stage 2. The env is `FrankaDroidEnv` -- a real
  7-DoF Franka arm with differential IK, contacts and a Robotiq gripper (grasping is
  physical), so Stage 1 exercises real arm dynamics, not a kinematic mocap toy. Embodiment
  note: Franka in sim (matches V-JEPA's DROID training); the physical target is a UR7e
  (Stage 2). See [`DESIGN.md`](DESIGN.md#embodiment-franka-in-sim-ur7e-on-hardware).
- **Stage 2.** The same orchestrator on the physical UR7e for a single vial-placement
  task: real coarse-placement error, end-to-end seat success, and a setup-cost comparison
  against classical calibration on real holders.
- **Stage 3 (grand plan).** Navigation between stations, real instruments, and more
  chemistry and labware.

## Immediate next steps

Reproduction is complete: the encoder + AC predictor load, CEM latency is characterized (~32 s at
800 samples), the camera ablation fixed a view-relative action frame (`PLANNING_CAMERA` = az45_el45),
DROID transition scoring gives a real-robot baseline (rank_frac 0.820 vs 0.486 null), and closed-loop
CEM reaches goal images. The fixed-bundle benchmark is built and running.

1. **Full benchmark (running).** 5 tasks x cup/box x 50 = 500 rollouts at paper settings (800
   samples, horizon 1, maxnorm 0.075, camera B_closer). Live results in
   `results/benchmarks/full800_B_progress/`; demo reel in `results/demos/full800_B/`. Publish the
   final per-(task, object) precision curves against paper Table 3.
2. **W* calibration.** Fit and apply the App. B.4 horizontal rotation so a non-az45 camera (e.g. a
   DROID-like view) can be used without breaking the action frame; the camera A/B/C experiment showed
   an azimuth change collapses planning without it.
3. **Predictor fine-tuning + re-benchmark.** Fine-tune the predictor (frozen encoder) on small task
   data; re-run transition scoring and the closed-loop benchmark at the same settings and report the
   deltas.
4. **ManiSkill (Linux/WSL2).** Adapter render -> V-JEPA latent -> CEM -> official success on
   PickCube / StackCube / PegInsertionSide.
5. **Confidence-gate data.** Perturb the start pose, plan, log terminal energy + baselines + the
   success label — the project's central measurement.

## Stage-1 build checklist

- [x] Reproducible local environment (venv + CUDA Torch + MuJoCo), verified on the 3090.
- [x] MuJoCo Franka scene and `FrankaDroidEnv` (render, real 7-DoF EE, goal capture). The
      earlier kinematic `MujocoPilotEnv` was removed in the clean-slate reset (recoverable from git
      history).
- [x] V-JEPA 2-AC interface scaffold and download-only checkpoint fetch.
- [x] Test suite (geometry, Franka env, grasp physics, success, utils) passing.
- [x] Franka Panda (MuJoCo Menagerie) loaded, rendered, actuated, and timed (smoke test).
- [x] Franka + Robotiq 2F-85 composed (mjSpec), exocentric camera, EE-space control via
      differential IK (`FrankaDroidEnv`); scripted reach test passes 5/5.
- [x] `apply_action` dynamically stepped (IK -> ctrl -> mj_step) with a measured gripper
      opening and action bounds (commit abeaad6; the substrate for grasp / world-model wiring).
- [x] Encoder + AC predictor load from the local checkpoint; encoder-only + CEM inference
      run in `scripts/vjepa2_ac_infer_test.py`.
- [x] CEM latency measured (V-JEPA 2-AC, bf16 on the 3090: 800 samples = 32 s, chunked).
- [x] CEM planning to a rendered goal image, in the env loop (`scripts/cem_reach_loop.py`):
      reach succeeds; multi-goal chaining works. Interface calibration (W*) still pending.
- [x] Graspable-object env substrate: cube + place zone + hidden success functions
      (`src/bench/success.py`), scripted grasp-lift regression passes.
- [x] Closed-loop task-success benchmark runner with multi-threshold precision curves,
      paper-faithful pick_place (4/10/4), CEM chunking, and a GT-vs-V-JEPA side-by-side demo.
- [x] **Fixed-bundle rebuild (objects + generator):** cup + box objects (+ distractors) in
      `franka_build.py` (cup = cube cup/open-top square box; one-wall rim grasp with one finger
      inside and one outside); `scripts/generate_task_bundles.py` scripted-expert generator with
      randomized arm+object starts, wide variety, and a mocap place zone randomized per pick_place
      trial; **400 bundles** under `tasks/` (grasp/reach_with_object/grasp_and_reach/pick_place x
      cup/box x 50, 0 skipped), inspected + approved via `scripts/inspect_task_viewer.py` (live
      MuJoCo window, N/B to step stages).
- [x] **`--bundles` loader:** `run_closed_loop_benchmark.py --bundles tasks` LOADS the fixed bundles
      (`env.set_state` restores qpos0 + zone, honors `start_grasped`, plans to the saved goal/sub-goal
      images, auto-switches sub-goals on the step budget, scripts only the gripper, scores
      object-in-zone + released with the swept sphere `x`). Smoke-validated end-to-end; new task
      thresholds/gates in `src/bench/thresholds.py`.
- [x] **Audit fixes (P1/P2) + docs sync:** CLI mode guard, error-aware `bundle_classify`, THRESHOLDS
      as single source of truth, placement-fair `object_placed`, provenance (mode + bundle_id), fail-loud
      loader, report GIFs, viewer/schema/threshold docs. Tests: `tests/test_bundle_bench.py` (49 passed).
      Held-grip stability empirically verified (no physics change). Committed + pushed to main (8c129aa).
- [x] **place_with_object task + camera experiment:** added `place_with_object` (the place half of
      pick_place; object starts held, 100 bundles). Camera A/B/C experiment: B_closer (same angle,
      closer) modestly improves fine positioning; a DROID-like angle collapses planning without W*.
      Diagnostic tooling: `--plan-gripper`, `replay_from_log.py` (reproduce any trial in 3D from the
      log), `make_demo_gifs.py` (labeled HIT/MISS reel).
- [~] **Full run (RUNNING):** 5 tasks x cup/box x 50 = 500 rollouts at samples 800, horizon 1,
      maxnorm 0.075, camera B_closer. So far: reach_with_object 98%/96% (beats paper 75%), grasp
      cup 40%/box 12%, grasp_and_reach/cup 31%; pick_place + place_with_object pending. Live report
      `results/benchmarks/full800_B_progress/`, demos `results/demos/full800_B/`. Halt via
      `logs/full_bench/STOP`; resume via `logs/full_bench/resume_full800_B.ps1`.
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
