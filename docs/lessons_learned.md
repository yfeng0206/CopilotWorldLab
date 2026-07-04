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
  function arguments`. Re-confirmed 2026-07-04: it fails even with the BASIC (non-OSC)
  controller, so it is in the core dynamics path, not just the OSC controller. robosuite
  1.5.2 is the latest release and still calls the old 2-arg `mj_fullM`; mujoco 3.10 requires
  `mj_fullM(model, data, dst)`.
- **Rule**: robosuite needs an older mujoco (~3.2.x). Our stack pins mujoco 3.10 for
  `FrankaDroidEnv` (the mjSpec composition API), so downgrading is not an option. Decision:
  do NOT adopt robosuite in the main env; if a standard third-party benchmark is ever needed,
  create a SEPARATE venv pinned to a compatible mujoco. It is not required for this project --
  robosuite is not what V-JEPA 2-AC uses (real Franka/DROID hardware), and `FrankaDroidEnv`
  already reproduces that hardware faithfully.

### 12. Differential IK returns a stale residual unless recomputed after the loop
- **What happens**: The residual is computed at the top of each iteration, before the last
  joint update; on a non-converged solve the returned error is one step stale.
- **Rule**: After the IK loop, recompute forward kinematics and the residual for the final
  configuration before returning it. Callers use the residual to detect failed solves.

### 13. Large model weights live on D: via a directory junction
- **What happens**: C: (231 GB) fills up with the 11 GB checkpoint plus caches; D: has
  1.3 TB free.
- **Rule**: The checkpoint lives at `D:\CopilotWorldLab\checkpoints\`; the repo's
  `checkpoints/` is a Windows junction pointing there, so relative paths like
  `checkpoints/vjepa2-ac-vitg.pt` work unchanged. Future model downloads go to D: via
  `TORCH_HOME` / `HF_HOME` (set to `D:\CopilotWorldLab\cache\...`). Do not commit the
  junction target; `checkpoints/` is already gitignored.

---

## V-JEPA 2-AC inference (CEM planning)

### 14. fp32 disables flash attention; always plan in bf16
- **What happens**: Running the ViT-g predictor in fp32 is several times slower than bf16
  for the same CEM config.
- **Why**: torch's scaled-dot-product-attention only dispatches the fused flash /
  mem-efficient kernels for fp16/bf16; fp32 falls back to the slow math kernel.
- **Rule**: Wrap encode + CEM in `torch.autocast(device, dtype=torch.bfloat16)`. bf16 is
  the intended inference precision for this model.

### 15. High CEM sample counts hit a CUDA allocator cliff; chunk the predictor batch
- **What happens**: bf16 timing was linear up to 400 samples (~0.04 s/sample) then jumped
  from an expected ~32 s to **148 s** at 800 samples on the 24 GB 3090. A predictor-vs-pose
  breakdown showed the predictor was still only 31 s and the CPU pose update 1.2 s -- ~115 s
  was framework overhead that appeared exactly when peak memory crossed ~12 -> 17 GiB.
- **Why**: once the activation working set is large enough that the allocator cache cannot
  serve a request, PyTorch falls back to synchronous `cudaMalloc`/`cudaFree` per step
  (allocator thrash). It is a memory-pressure artifact, not extra compute.
- **Rule**: chunk the CEM sample batch through the predictor (`--chunk`, default 200) so
  peak stays in the linear regime. Chunking is numerically identical (each sample is an
  independent batch row) -- verified the planned action is unchanged. Result: 800 samples
  drop to 32 s (predictor-bound) at 15 GiB peak. Do NOT set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on Windows; it is unsupported there and
  only prints a warning.

---

## Simulation and world-model interface

### 16. `mj_forward` does not integrate actuators; step physics to preview a driven gripper
- **What happens**: `capture_goal_image` set the Robotiq gripper command and called
  `mj_forward`, but open- and closed-gripper goal images rendered identically.
- **Why**: `mj_forward` computes kinematics/dynamics for the current state; it does not
  advance actuators toward their targets. The 2F-85 fingers are a driven linkage, so their
  qpos only moves when the physics is stepped.
- **Rule**: to preview any position-actuated part (the gripper), briefly `mj_step` with the
  command set (save and restore `qpos`/`qvel`/`ctrl`), do not rely on `mj_forward` after
  writing `data.ctrl`. Covered by `test_goal_image_reflects_gripper_state`.

### 17. The V-JEPA CEM action bound is a per-axis box, and our env clamps the L2 norm
- **What happens**: docs called `maxnorm = 0.075` an "L1 ball". It is not.
- **Why**: the reference CEM clips each translation axis independently to
  `[-maxnorm, maxnorm]` (an axis-aligned box / L-inf ball, ~13 cm Euclidean diagonal) and
  zeros rotation, whereas `FrankaDroidEnv` bounds the L2 norm of the translation to
  `max_translation` (0.13 m) -- a different constraint shape.
- **Rule**: describe the CEM bound as a box, not an L1 ball, and reconcile the box (CEM) vs.
  L2 (env) translation limits during interface calibration before any zero-shot transfer.

### 18. ManiSkill (SAPIEN) does not run on this Windows setup
- **What happens**: `mani-skill` 3.0.1 + `sapien` 3.0.3 install fine in a separate venv, but
  `PickCube-v1` fails: the `pd_ee_delta_pose` (end-effector) control mode raises
  `TypeError: 'NoneType' object is not callable` because `PinocchioModel` is `None` (Pinocchio,
  the IK backend, has no Windows wheel), and even the `pd_joint_delta_pos` control mode crashes
  the process with an access violation (exit `0xC0000005`) inside the SAPIEN native sim.
- **Why**: SAPIEN's GPU/physx simulator and the Pinocchio IK dependency are Linux-oriented; the
  Windows path is not reliable. Same class of blocker as robosuite (#11).
- **Rule**: the two standard sim manipulation suites (robosuite, ManiSkill) both require
  Linux / WSL2 for the closed-loop success benchmark. On Windows, use the MuJoCo
  `FrankaDroidEnv` (which works) as the closed-loop platform, or move ManiSkill to WSL2/Linux.
  The V-JEPA scoring/planning code is backend-agnostic, so only the env adapter changes.
