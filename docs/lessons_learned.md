# Lessons Learned

Mistakes, debug-traps, and invariants we have paid to learn. Keep them visible so they do
not sneak back in.

---

## Environment (Windows + CUDA)

### 1. Win32 VRAM readout is wrong; trust nvidia-smi
- **What happens**: `Get-CimInstance Win32_VideoController` reports the RTX 3090 as
  having 4 GB of VRAM.
- **Why**: The WMI `AdapterRAM` field is a 32-bit value and saturates at 4 GB; it does
  not reflect real VRAM on modern GPUs.
- **Rule**: Use `nvidia-smi --query-gpu=memory.total --format=csv`. The 3090 has 24 GB.

### 2. Windows ships a fake `python`
- **What happens**: `python` resolves to the Microsoft Store app-execution alias and
  prints "Python was not found", even though scripts expect a real interpreter.
- **Rule**: Install a real Python (winget `Python.Python.3.11`, user scope) and call the
  venv interpreter by full path in scripts; do not rely on the bare `python` alias.

### 3. PyTorch must come from the CUDA wheel index on Windows
- **What happens**: `pip install torch` from PyPI gives a CPU-only build; `cuda.is_available()`
  is False and nothing runs on the 3090.
- **Rule**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
  Verified: torch 2.6.0+cu124 sees the 3090.

### 4. MuJoCo rendering on Windows is WGL-only and needs a desktop
- **What happens**: EGL/OSMesa headless recipes copied from Linux fail on Windows.
- **Why**: MuJoCo's GL backend only lists `wgl` for Windows; EGL and OSMesa are
  Linux-only. WGL needs an interactive window station / desktop session.
- **Rule**: On Windows leave `MUJOCO_GL` unset (defaults to wgl) and run from an
  interactive desktop. On a headless Linux box, set `MUJOCO_GL=egl`.

### 5. Render tests must skip, not fail, when there is no GL context
- **What happens**: A CI or service box with no OpenGL context makes every render test
  hard-fail, masking real regressions.
- **Rule**: Split tests -- model + kinematics run without GL; render tests attempt one
  render and `pytest.skip` if the failure looks like a GL/context error, else re-raise.
  See `tests/test_render.py`.

---

## Models and data

### 6. The V-JEPA 2-AC checkpoint is 11 GB and not on HuggingFace
- **What happens**: Searching HuggingFace for `vjepa2-ac` returns only encoder/probe
  checkpoints, not the action-conditioned model.
- **Why**: The AC model ships as a single `.pt` on `dl.fbaipublicfiles.com` containing
  both `encoder` and `predictor` state dicts; HuggingFace hosts encoders only.
- **Rule**: Download the AC `.pt` directly (resumable), or load via
  `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')`. Budget ~11 GB.

### 7. Verify citations against full text, not abstracts
- **What happens**: Earlier we "fixed" the DreamTacVLA / AHEAD citations as wrong based on
  abstracts; a full-text read showed the original characterisations were correct and the
  "fix" was the error.
- **Rule**: Read the actual paper (arXiv HTML/PDF) before characterising a method. Treat
  generic LLM-summary web search as unreliable -- it hallucinates titles and methods.

---

## Repo hygiene

### 8. Keep secrets and large/generated artifacts out of the tree
- **Rule**: `.gitignore` covers `*token*.txt`, `checkpoints/`, `data/`, `.venv/`,
  `third_party/`, `*.docx`, the proposal workflow (`scripts/proposal/`, `fig_*.png`) and
  the local `CHANGELOG.md`. The committed tree is code + docs only.

---

## Simulation and control (Franka / robosuite)

### 9. The end-effector site must be the gripper TCP, not the arm flange
- **What happens**: Attaching Robotiq to the Panda leaves the arm's `attachment_site`
  (flange) at ~15.6 cm above the actual grasp point. Controlling/measuring the flange
  makes wrist rotations move the grasp point incorrectly.
- **Rule**: Use the gripper TCP (`2f85_pinch`) as the EE site for control and state. Keep
  `attachment_site` only as the mount point for the spec merge.

### 10. mjSpec merge drops the child's physics options; re-set them after merging
- **What happens**: Attaching `robotiq_2f85` (elliptic friction cone, impratio 10) onto the
  Panda keeps the parent's pyramidal cone; a warning is printed and the gripper's intended
  contact model is silently lost.
- **Rule**: After `attach_body`, explicitly set `spec.option.cone = mjCONE_ELLIPTIC` and
  `spec.option.impratio` before `compile()`. Verify on the compiled model (`m.opt.cone`).

### 11. robosuite 1.5.2 is incompatible with mujoco 3.10
- **What happens**: `robosuite.make(...).reset()` raises `TypeError: mj_fullM(): incompatible
  function arguments` -- robosuite's OSC controller calls `mj_fullM(model, dst, qM)` (old
  2-arg-ish form) but mujoco 3.10 requires `mj_fullM(model, data, dst)`.
- **Rule**: robosuite needs a mujoco version matching its release (~3.3.x). Our stack pins
  mujoco 3.10 (torch + our env), so we use our own Franka+IK env; only adopt robosuite in a
  separate env or once it supports 3.10.

### 12. Differential IK returns a stale residual unless recomputed after the loop
- **What happens**: The residual is computed at the top of each iteration, before the last
  joint update; on a non-converged solve the returned error is one step stale.
- **Rule**: After the IK loop, recompute forward kinematics and the residual for the final
  configuration before returning it. Callers use the residual to detect failed solves.
