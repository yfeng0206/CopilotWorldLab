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

Stage-1 pilot, set-up phase. This session established a reproducible local environment
and a minimal MuJoCo scaffold that exposes exactly the interface the world model will
consume. No world-model inference has been run yet.

Working now (verified on Windows 11 + RTX 3090, CUDA 12.4, 15/15 tests passing):

- A minimal MuJoCo tabletop scene (vial, holder well, mocap end-effector, two cameras).
- `MujocoPilotEnv`: headless RGB rendering, a 7-DoF end-effector state/action interface
  matching the V-JEPA 2-AC 7-DoF layout, and goal-image capture.
- A download-only checkpoint fetcher and a one-command environment setup script.
- A V-JEPA 2-AC wrapper scaffold that records the verified inference interface but runs
  no network (wired up next session).

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
