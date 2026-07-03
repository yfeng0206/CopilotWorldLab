# Architecture

Component-level specification for the Stage-1 pilot. This documents the interfaces that
are implemented today (the MuJoCo environment) and the interface the world model will
plug into (the V-JEPA 2-AC wrapper), so the two halves can be developed against a stable
contract.

## 1. The 7-DoF end-effector interface (the contract)

Everything is organised around a single 7-D end-effector vector, chosen to match V-JEPA
2-AC's action *layout* (arXiv:2506.09985, Section 3.1):

```
state / action layout:  [ x, y, z, roll, pitch, yaw, gripper ]
                          <-- position -->  <- extrinsic XYZ ->  <- [0,1] -->
```

- Position is metres in the world frame.
- Orientation is extrinsic XYZ Euler angles (radians). See `src/utils/geometry.py` for
  the Euler <-> quaternion conversions and their unit tests.
- Gripper is a scalar in `[0, 1]` (0 = open, 1 = closed).
- An *action* is a delta on this vector between consecutive frames (V-JEPA 2-AC:
  "the change in end-effector state between frames k and k+1").

### Interface-distribution risks (validate before trusting zero-shot planning)

Matching the 7-D *layout* is necessary but not sufficient. V-JEPA 2-AC was trained on real
Franka DROID trajectories with a particular camera viewpoint, action cadence and action
distribution, and it must infer the action coordinate axis from the monocular RGB image.
Before interpreting any planning result, calibrate the interface:

- **Frame / sign / scale.** Confirm world- vs body-frame deltas, per-axis signs, and
  translation/rotation scale by scoring known sim transitions `(x_k, s_k, a_gt, x_{k+1})`
  and checking that `a_gt` sits near the latent-energy minimum; sweep variants otherwise.
- **Camera.** Prefer a DROID-like third-person view (`scene_cam`) where the workspace axes
  and gripper motion are visible; treat the downward `wrist_cam` as a later ablation, not
  the default evidence source.
- **Cadence.** One action should represent the paper's ~4 fps (~0.25 s) interval, not a
  single 10 ms sim step; interpolate the mocap motion over the correct number of steps.
- **Dynamics.** The mocap end-effector is kinematic (no arm, IK, contact or finger
  articulation), so Stage 1 validates the latent scoring / planning interface in a toy
  renderer, not real robot-arm manipulation.

## 2. `MujocoPilotEnv` (`src/envs/mujoco_scene.py`)

A thin wrapper over the MJCF scene that exposes exactly what the world model needs. The
end-effector is a kinematic MuJoCo mocap body, so Stage 1 needs no arm, no IK and no
actuators; the only dynamic body is the vial.

| Method | Signature | Purpose |
|---|---|---|
| `render` | `(camera=None, width=None, height=None) -> uint8[H,W,3]` | RGB observation / goal image from a named camera |
| `get_ee_state` | `() -> float32[7]` | current 7-D end-effector state |
| `set_ee_pose` | `(pos=None, euler=None, quat=None, gripper=None)` | absolute pose set |
| `apply_action` | `(delta[7], step_physics=True, frame="world") -> float32[7]` | apply a 7-D delta, step physics, return new state |
| `capture_goal_image` | `(pos=None, euler=None, gripper=None, camera=None) -> uint8[H,W,3]` | render at a hypothetical pose, then restore state |
| `get_observation` | `(camera=None) -> {"image", "ee_state"}` | combined observation |
| `reset` / `close` | | reset to home pose / release the renderer |

Notes:
- `render` defaults to 256x256, the resolution V-JEPA 2 expects. A one-off different size
  spins up a temporary renderer so the cached one keeps the configured size.
- `capture_goal_image` saves and restores mocap pose + gripper, so goal rendering never
  perturbs the live state -- this is how goal latents `z_g` will be produced.
- Orientation deltas compose in the world frame by default (`frame="world"`); body-frame
  composition is available for wrist-relative motion.

### Scene (`assets/mujoco/scene.xml`)

- A table (top surface at z = 0.22 m).
- A fixed square holder "well" with an inner opening ~0.04 m across; the vial is 0.024 m
  in diameter, giving a generous few-millimetre clearance for the initial task (tightened
  in later tests, matching the proposal's loosened-clearance plan).
- A free-jointed vial (capped cylinder), the only dynamic body.
- A mocap end-effector proxy (palm + two finger geoms) carrying a downward `wrist_cam`.
- A fixed `scene_cam` aimed at the workspace.

## 3. `VJEPA2ACWorldModel` (`src/world_model/vjepa2_wrapper.py`)

The control-loop wrapper is still a thin scaffold, but the model now loads and runs: a
working local-checkpoint loader and CEM-MPC timing harness live in
`scripts/vjepa2_ac_infer_test.py` (see `docs/setup_stage.md` for the measured timings).
The wrapper defines the methods the control loop will call:

| Method | Purpose | Status |
|---|---|---|
| `encode(frames)` | frames -> latent `z` (frozen ViT-g, per-frame 16x16x1408) | in harness |
| `predict(latent, state, actions)` | action-conditioned rollout `P(a; s, z)` | in harness |
| `latent_energy(pred, goal)` | `mean(|pred - goal|)` = the gate signal | implemented (pure array op) |
| `plan_action(obs, goal_image, config)` | CEM MPC to a goal image | in harness (not yet in the env loop) |
| `load_vjepa2_ac(device, source)` | local-checkpoint loader | wired in the harness; wrapper method still a stub |

Working loading path (implemented in the harness; not `torch.hub`, whose base URL in the
vendored repo is a localhost stub):

```python
import torch
from src.hub.backbones import _make_vjepa2_ac_model, _clean_backbone_key  # vendored repo
encoder, predictor = _make_vjepa2_ac_model("vit_ac_giant", pretrained=False)
state = torch.load("checkpoints/vjepa2-ac-vitg.pt", map_location="cpu", weights_only=True)
encoder.load_state_dict(_clean_backbone_key(state["encoder"]), strict=False)   # RoPE-tolerant
predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=True)
```

## 4. Planner (`PlannerConfig`)

Cross-Entropy-Method MPC; defaults follow the V-JEPA 2-AC paper (Table 3):

| Field | Default | Source |
|---|---|---|
| `samples` (population) | 800 | paper Table 3 |
| `iterations` | 10 | paper Table 3 |
| `horizon` | 1 | paper (greedy one-step + receding horizon) |
| `top_k` | 10 | our default (paper: "top-k", exact k in appendix) |

Additional replication knobs (verified against the reference CEM): each translation axis
is sampled and clipped independently to `[-0.075, 0.075]` m -- an axis-aligned box (L-inf
ball), not an L1 ball, so up to ~13 cm Euclidean displacement per action; rotation is
zeroed and the gripper is sampled. 4 fps action rate, 256x256 input. Reported latency
~16 s/action on an RTX 4090.

## 5. The confidence gate

The gate consumes `latent_energy(pred, goal)` (Option 1) or ensemble disagreement
(Option 2) and decides handoff B -> C. Threshold fit on a held-out calibration set to a
target false-accept rate; below threshold, bounded retry then flag a human. The gate is
the project's central measurement (ROC AUC vs a simple baseline and vs the visual servo's
pixel-error convergence). Not implemented yet; the pilot collects the trials that will
fit and evaluate it.

## 6. Hardware envelope (local)

- Windows 11, RTX 3090 (24 GB, ~23.5 GB free), driver 610.62 / CUDA 13.3, 32 GB RAM.
- Sufficient for ViT-g inference and CEM planning locally. A single A100-class GPU is
  needed only for later fine-tuning, not for the off-the-shelf pilot.
- Rendering uses the WGL/GLFW backend (the only Windows option; EGL/OSMesa are
  Linux-only) and needs an interactive desktop session.
