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
