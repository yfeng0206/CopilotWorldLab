# CopilotWorldLab

Learned world-model manipulation for self-driving chemistry laboratories: a latent
video world model drives the coarse arm motion, and that same model's own predictive
energy is proposed as the gate that hands off to a classical, vision-only precise seat.

This repository is the engineering workspace for the pilot. The formal proposal prose
lives in the design docs below; the proposal document generator and its figures are
kept locally and are not part of the tracked tree.

## Idea

Self-driving ("autonomous") chemistry labs already automate experiment selection, but
the robot arm is still run by classical, pre-programmed control that needs per-station
calibration and custom labware. This project adds two learned layers on top of an
otherwise standard lab:

1. An LLM planner that turns a natural-language request into a schedule of typed
   machine actions and re-plans as results arrive.
2. A world-model arm controller that performs the variable part of arm motion -- the
   coarse approach to and placement of labware at an instrument.

Classical control is kept for the precise and safety-critical steps. A latent world
model (V-JEPA 2-AC) drives the coarse approach by planning to a goal image; a
vision-only visual servo performs the sub-millimetre seat; and the world model's own
predictive energy is proposed as the competence gate that decides when to hand off from
the learned coarse stage to the precise stage. Whether that confidence signal reliably
predicts a failed handoff is the project's central open question.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design and the honest novelty claim,
and [`docs/related_work.md`](docs/related_work.md) for full-text-verified notes on the
closest prior art.

## Status

Stage-1 pilot, pre-experiment setup stage -- complete. This session established a
reproducible local environment, a DROID-style Franka arm in MuJoCo, verified loading of
the V-JEPA 2-AC checkpoint, and a characterized inference/timing baseline on the 24 GB
RTX 3090. No closed-loop world-model control has been run yet; that is the first
experiment. See [`docs/setup_stage.md`](docs/setup_stage.md) for the full record (setup
milestones, the timing table, and the audit/cleanup log).

Working now (verified on Windows 11 + RTX 3090, CUDA 12.4, 25/25 tests passing):

- A DROID-style Franka Panda + Robotiq 2F-85 MuJoCo env (`FrankaDroidEnv`) with dynamic
  7-DoF end-effector control matching the V-JEPA 2-AC action layout.
- Local V-JEPA 2-AC loading (ViT-g encoder 1.01B + AC predictor 305M) and a CEM-MPC
  inference/timing harness: 800-sample planning in 32 s at 15 GiB (bf16, chunked),
  consistent with the paper's 16 s on a ~1.8x-faster 4090.
- JEPA-style logging, a download-only checkpoint fetcher, and a one-command setup script.

## Repository layout

```
assets/mujoco/scene.xml         Minimal tabletop MJCF (vial, holder, mocap EE, cameras)
configs/mujoco_pilot.yaml        Step-1 scene / render / action / planner knobs
src/envs/mujoco_scene.py         MujocoPilotEnv: render + 7-DoF EE + goal capture
src/world_model/vjepa2_wrapper.py   V-JEPA 2-AC interface scaffold (no inference)
src/utils/geometry.py            SO(3) helpers (extrinsic-XYZ Euler <-> quaternion)
src/utils/config.py              Tiny YAML config loader
scripts/setup_env.ps1            Reproducible venv + CUDA Torch + deps
scripts/download_checkpoints.py  Download V-JEPA 2 weights (download only)
tests/                           Geometry, env-kinematics, and render tests
docs/                            DESIGN, architecture, related_work, research_log, ...
```

## Setup

Verified on Windows 11, RTX 3090 (24 GB), driver 610.62 / CUDA 13.3, Python 3.11.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\Activate.ps1
python -m pytest -q
```

PyTorch is installed from the CUDA 12.4 wheel index; the rest come from
`requirements.txt` (or use `environment.yml` for conda). To fetch the model weights
(download only, no inference):

```powershell
python scripts\download_checkpoints.py            # action-conditioned checkpoint
python scripts\download_checkpoints.py --encoder vitl   # optional encoder-only
```

## Roadmap

- Stage 1 (in progress): the software pilot -- MuJoCo scene, world-model coarse
  placement, and the confidence-gate measurement, evaluated in simulation.
- Stage 2: the same orchestrator on the physical UR7e for a single vial-placement task.
- Stage 3 (grand plan): navigation between stations, real instruments, more labware.

Immediate next steps and the full backlog live in [`docs/plan.md`](docs/plan.md).

## References

- Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction
  and Planning.* arXiv:2506.09985 (2025). [link](https://arxiv.org/abs/2506.09985)
- Sun et al. *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model.*
  arXiv:2602.10098 (2026). [link](https://arxiv.org/abs/2602.10098)
- Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246
  (2024). [link](https://arxiv.org/abs/2406.09246)
- Syed et al. *Intercepting the Future: Latent-Space Predictive World Model for Dynamic
  VLA Manipulation* (AHEAD). arXiv:2606.02486 (2026).
  [link](https://arxiv.org/abs/2606.02486)
- Ye et al. *Learning to Feel the Future: DreamTacVLA for Contact-Rich Manipulation.*
  arXiv:2512.23864 (2025). [link](https://arxiv.org/abs/2512.23864)
- Yu et al. *Siamese Convolutional Neural Network for Sub-millimetre-accurate Camera Pose
  Estimation and Visual Servoing.* arXiv:1903.04713 (2019).
  [link](https://arxiv.org/abs/1903.04713)
- Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.*
  arXiv:2403.12945 (2024). [link](https://arxiv.org/abs/2403.12945)
- Todorov et al. *MuJoCo: A physics engine for model-based control.* IROS 2012;
  [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco).

A full, context-annotated bibliography is in
[`docs/research_log.md`](docs/research_log.md#paper-bibliography).
