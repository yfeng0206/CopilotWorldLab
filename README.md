# CopilotWorldLab

Learned world-model manipulation for self-driving chemistry laboratories: a latent video world
model (V-JEPA 2-AC) drives the coarse robot-arm motion by planning to a goal image, and that same
model's own predictive energy is proposed as the gate that hands off to a classical, vision-only
precise seat. This repository is the **Stage-1 simulation** substrate; the design rationale and the
broader proposal live in [`docs/DESIGN.md`](docs/DESIGN.md).

## Current phase

**Phase 0 — set up and reproduce V-JEPA 2-AC honestly.** Load the released model, build a
paper-faithful MuJoCo env, and verify the world model reproduces the paper (energy landscape,
transition scoring on real DROID, closed-loop CEM to goal images) before adding our own method.
The full roadmap is in [`docs/DESIGN.md`](docs/DESIGN.md#0-project-roadmap-phases):

| Phase | What | State |
|---|---|---|
| **0** | Setup + reproduce V-JEPA 2-AC world model | done |
| **1** | Fixed-bundle closed-loop benchmark: Reach / Grasp / Reach-with-object / Pick-Place x cup/box, 50 trials each | current |
| 2 | POV/wrist CNN coarse-to-fine (improvement #1) | planned |
| 3 | 3rd + first-person cross-attention latent (our method) | planned |
| 4 | Unified cross-view latent | planned |

## Results so far

Reproducible experiments with honest, primary-source-verified numbers (see
[`docs/experiments/`](docs/experiments)):

- **Energy-landscape reproduction** (correctness gate): the model reproduces the paper's behaviour
  — latent-energy minimum near the ground-truth action (reverse cos **+0.98**), reverse flips.
- **Camera-placement ablation**: the horizontal action frame is view-relative; the best zero-shot
  view (`az45_el45`, now `PLANNING_CAMERA`) improves action-alignment cosine by **+1.08** over the
  built-in camera. [writeup](docs/experiments/energy_landscape_and_camera_ablation.md)
- **Transition scoring on real DROID** (vanilla baseline, n=300 from `lerobot/droid_100`): the true
  action beats **82.0%** of random negatives vs a **48.6%** (chance) different-episode null — a
  +0.334 image-conditioning effect. The fine-tuned predictor will be measured against this.
  [writeup](docs/experiments/transition_scoring.md)
- **Closed-loop CEM (Phase 1 pilot)**: reach to a goal image **succeeds** in the control loop;
  the ~3 cm precision floor is diagnosed as a model/interface limit (tracking error only 9 mm).
  [writeup](docs/experiments/cem_closed_loop.md)
- **Closed-loop task success (Phase 1, in rebuild)**: honest **Reach / Grasp / Reach-with-object /
  Pick-Place** (the paper's four robot tasks, arXiv 2506.09985 Table 3) on our own MuJoCo env —
  V-JEPA plans the coarse motion, scripted gripper, hidden-state success. Each task runs on **two
  objects** (a rim-graspable **cup** and a rigid **box**) on the **same** scene with the target geom
  swapped, and on **fixed, saved task bundles** (start + sub-goal + goal frames + states + camera per
  scenario, inspectable under `tasks/`), so runs are reproducible and every config is scored on
  identical scenarios. Success = Euclidean delta within a **swept sphere radius `x`** (mean delta +
  success rate reported per `x`). **50 trials per (task, object)** = 400 scenarios. Methodology:
  [closed_loop_benchmark.md](docs/experiments/closed_loop_benchmark.md).

Honest boundary: only Reach is a pure V-JEPA success; grasp/pick-place are V-JEPA coarse motion +
scripted gripper, scored on hidden privileged sim state. The earlier random-per-trial runs (and their
committed reports) were cleared in a clean-slate reset to move to the fixed-bundle + two-object design
above; they remain recoverable from git history.

## Repository layout

```
src/envs/franka_build.py         Compose Franka Panda + Robotiq 2F-85 (+ cup/box object, place zone, distractors)
src/envs/franka_droid_env.py     FrankaDroidEnv: real 7-DoF EE control via differential IK + physics
src/bench/schema.py              Task-bundle schema (start/goal/sub-goal images + states + model XML)
src/bench/success.py             Hidden success functions (reach / touch / grasp / place)
src/bench/thresholds.py          Precision thresholds + physical gate spec per task
src/envs/robomimic_render.py     Render robomimic raw demos on Windows (reference/image source only)
src/utils/{ik,geometry,logging,config}.py   IK, SO(3) helpers, JEPA-style logging, YAML config
src/world_model/vjepa2_wrapper.py            V-JEPA 2-AC control-loop scaffold
scripts/download_checkpoints.py  Fetch V-JEPA 2 weights (size + SHA256 verified)
scripts/vjepa2_ac_infer_test.py  Load V-JEPA 2-AC, time CEM planning (bf16, chunked)
scripts/generate_task_bundles.py Scripted expert -> fixed task bundles under tasks/ (start/sub-goals/goal + states)
scripts/run_closed_loop_benchmark.py  Closed-loop CEM-MPC benchmark; loads fixed bundles, hidden-state success
scripts/energy_landscape_repro.py, render_franka_transitions.py, analyze_frame_rotation.py
scripts/benchmark_transition_scoring.py, extract_droid_transitions.py, plot_transition_benchmark.py
scripts/cem_reach_loop.py, plot_cem_loop.py     Closed-loop CEM planning to goal image(s)
tasks/                           Fixed, inspectable task bundles (reach/grasp/pick_place x cup/box)
tests/                           Geometry, Franka env, grasp physics, success, thresholds, utils
docs/                            architecture, DESIGN, experiments/, research_log, lessons_learned, ...
```

## Setup

Verified on Windows 11, RTX 3090 (24 GB), Python 3.11. PyTorch is installed from the CUDA 12.4
wheel index; the rest from `requirements.txt` (or `environment.yml` for conda).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\Activate.ps1
python -m pytest -q
python scripts\download_checkpoints.py        # V-JEPA 2-AC checkpoint (~11.8 GB)
```

### Vendored third-party repos (gitignored; fetch once)

The V-JEPA scripts import `third_party/vjepa2`; the Franka env needs MuJoCo Menagerie:

```powershell
git clone https://github.com/facebookresearch/vjepa2 third_party/vjepa2
git -C third_party/vjepa2 checkout 204698b

git clone --depth 1 --filter=blob:none --sparse `
  https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda robotiq_2f85
```

## Links

| | |
|---|---|
| Design, novelty claim, roadmap | [`docs/DESIGN.md`](docs/DESIGN.md) |
| Technical architecture + flowcharts | [`docs/architecture.md`](docs/architecture.md) |
| Compute budget, checkpoint, fine-tune plan | [`docs/vjepa2_ac_architecture.md`](docs/vjepa2_ac_architecture.md) |
| Experiments (energy landscape, ablation, benchmarks, closed loop) | [`docs/experiments/`](docs/experiments) |
| Closed-loop task-success benchmark (setup, criteria, logging, results) | [`docs/experiments/closed_loop_benchmark.md`](docs/experiments/closed_loop_benchmark.md) |
| Evaluation strategy | [`docs/experiments/benchmark_plan.md`](docs/experiments/benchmark_plan.md) |
| Related work (full-text-verified) | [`docs/related_work.md`](docs/related_work.md) |
| Research log + bibliography | [`docs/research_log.md`](docs/research_log.md) |
| Lessons learned (debug traps, invariants) | [`docs/lessons_learned.md`](docs/lessons_learned.md) |
| Plan / backlog | [`docs/plan.md`](docs/plan.md) |

## References

- Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
  Planning.* arXiv:2506.09985 (2025). [link](https://arxiv.org/abs/2506.09985)
- Sun et al. *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model.*
  arXiv:2602.10098 (2026). [link](https://arxiv.org/abs/2602.10098)
- Syed et al. *Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA
  Manipulation* (AHEAD). arXiv:2606.02486 (2026). [link](https://arxiv.org/abs/2606.02486)
- Ye et al. *Learning to Feel the Future: DreamTacVLA for Contact-Rich Manipulation.*
  arXiv:2512.23864 (2025). [link](https://arxiv.org/abs/2512.23864)
- Yu et al. *Siamese CNN for Sub-millimetre-accurate Camera Pose Estimation and Visual Servoing.*
  arXiv:1903.04713 (2019). [link](https://arxiv.org/abs/1903.04713)
- Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.*
  arXiv:2403.12945 (2024). [link](https://arxiv.org/abs/2403.12945)
- Todorov et al. *MuJoCo: A physics engine for model-based control.* IROS 2012.

A full, context-annotated bibliography is in
[`docs/research_log.md`](docs/research_log.md#paper-bibliography).
