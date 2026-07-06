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

## Immediate next steps (next session)

Done so far: encoder + AC predictor load; CEM latency measured (~32 s at 800 samples);
interface calibration via the **camera ablation** (view-relative frame; `PLANNING_CAMERA` =
az45_el45 is the `FrankaDroidEnv` default); **benchmark 1** (transition scoring) hardened on a
**real DROID batch** (n=300 from `lerobot/droid_100`): rank_frac **0.820** vs different-episode
null **0.486** (+0.334 image-conditioning effect), top1 0.320, gap_z +1.45
(`docs/experiments/transition_scoring.md`); and **Phase 1 closed-loop CEM** to a goal image
(`docs/experiments/cem_closed_loop.md`) -- reach succeeds (goal image in 3 steps), multi-goal
chaining works, and the ~3 cm precision floor is diagnosed as a model/interface limit (tracking
error only 9 mm), motivating W* + fine-tuning. robosuite/ManiSkill closed-loop *rollout* is
Windows-blocked (lessons #11/#18), but robomimic raw states re-render on Windows for grasp/place
*task* sources (#11/#19); DROID gives the real-robot *transition* baseline. The plan is
benchmark-driven (`docs/experiments/benchmark_plan.md`). Remaining:

1. **Fixed-bundle closed-loop benchmark (Phase 1, in build).** Replace the random-per-trial scenarios
   with **fixed, saved task bundles** and run **grasp / reach_with_object / grasp_and_reach /
   pick_place** on **two objects** (a rim-graspable **cube cup** and a rigid **box**), **50 trials per
   (task, object)** = 400 scenarios, on **one env** with the target geom swapped. The set is inspired
   by the paper's robot tasks (arXiv 2506.09985 Table 3) but drops plain reach as uninteresting and
   adds the 2-goal grasp_and_reach composition. Success = Euclidean delta within a **swept sphere
   radius `x`** (mean delta + success@x). Steps: (a) add cup/box + distractors to `franka_build.py`;
   (b) `scripts/generate_task_bundles.py` — a scripted expert renders start/sub-goals/goal + states +
   camera per scenario into `tasks/…`, validated (expert must complete it) with contact sheets for
   inspection; (c) a `--bundles` loader in `run_closed_loop_benchmark.py` (deterministic, replaces
   `_rand_cube_xy`); (d) run at samples 200/400/800 and report per (task, object). Goal frames:
   grasp = 1 goal (object just grabbed, not lifted; scripted lift tests success); reach_with_object =
   1 goal (object starts in hand); grasp_and_reach = 2 goals (`goal_1` just grabbed, `goal` held-object
   target); pick_place = 3 goals (`goal_1`, `goal_2`, `goal`) on the fixed 4/10/4 schedule. No
   robomimic dependency. (`docs/experiments/closed_loop_benchmark.md`.)
2. **W* calibration + re-run.** Fit/freeze the App. B.4 horizontal rotation for the planning
   camera and re-run the benchmark; expect grasp/place error to drop -- the first improvement delta.
3. **FrankaDroidEnv closed-loop pick/place pilot** — DONE (early runner + hidden success).
4. **Predictor fine-tuning + re-benchmark.** Fine-tune the predictor (frozen encoder) on small
   task data; re-run benchmark 1 (same n/H/K/seed) + the closed-loop benchmark (same protocol) and
   report improvement as metric deltas vs the vanilla baselines.
5. **ManiSkill (benchmark 2, Linux/WSL2).** Adapter render -> V-JEPA latent -> CEM loop ->
   step -> official success on PickCube / StackCube / PegInsertionSide (gated on a Linux env).
5. **Confidence-gate data.** Perturb the start pose, plan, log terminal energy + confound
   baselines + success label (the project's central gate measurement).

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
- [ ] **`--bundles` loader:** make `run_closed_loop_benchmark.py` LOAD the fixed bundles (restore
      qpos0, use saved goal images, handle `start_grasped`) instead of `_rand_cube_xy`; add tests;
      then run at samples 200/400/800.
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
