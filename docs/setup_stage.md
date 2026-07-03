# Pre-Experiment Setup Stage (Stage 0)

Everything completed before the first real experiments (zero-shot V-JEPA 2-AC control in
MuJoCo, interface calibration, and predictor fine-tuning). This stage established a
reproducible local environment, a DROID-style Franka arm in simulation, verified loading
of the V-JEPA 2-AC checkpoint, JEPA-style logging, and a characterized inference/timing
baseline on the 24 GB RTX 3090. No closed-loop world-model control has been run yet; that
is the first experiment (see the roadmap at the bottom).

Verified on Windows 11, RTX 3090 (24 GB), CUDA 12.4 wheel, Python 3.11. Committed on branch
`world-model-pilot` (PR #1) as `yfeng0206`.

## 1. Setup milestones

| Area | What was done | Status | Evidence |
|---|---|---|---|
| Local environment | Python 3.11 `.venv`; torch 2.6.0+cu124 (CUDA verified on the 3090); mujoco 3.10.0 | Done | `scripts/setup_env.ps1`, `requirements.txt` |
| Repo restructure | Branch `world-model-pilot` mirroring I-JEPA_3D_OCT doc discipline; proposal generators kept local/gitignored | Done | PR #1 |
| GitHub identity | `gh` auth and all commits as `yfeng0206` (not an org identity) | Done | commits `abeaad6`, `edcf011` |
| Franka DROID env | Franka Panda + Robotiq 2F-85 composed via `mjSpec`; dynamic 7-DoF end-effector control (differential IK -> `ctrl` -> `mj_step`), measured gripper, TCP at `2f85_pinch` | Done | `src/envs/franka_build.py`, `src/envs/franka_droid_env.py`, `src/utils/ik.py` |
| Test suite | Model, kinematics, and render tests; render tests skip (not fail) without a GL context | Done (24 passed) | `tests/`, `pytest.ini` |
| V-JEPA 2-AC checkpoint | Verified public/ungated; downloaded 11 GB; relocated to `D:` via a directory junction | Done | `checkpoints/` junction; lessons #6, #13 |
| V-JEPA 2-AC loading | ViT-g encoder (1.01B) + AC predictor (305M) load from the local `.pt` with a fail-loud mismatch guard; namespace-isolated from the vendored `src` | Done | `scripts/vjepa2_ac_infer_test.py` |
| Logging | JEPA-style logging (`get_logger`, `CSVLogger`, `AverageMeter`, `gpu_timer`, `grad_logger`) mirrored from I-JEPA_3D_OCT | Done | `src/utils/logging.py` |
| Inference timing | CEM-MPC latency timed small -> big; found and fixed the 800-sample memory cliff (below) | Done | `logs/vjepa2_ac_timing_*.csv` |
| Compute / fine-tune plan | 24 GB memory budget; frozen-encoder predictor fine-tuning plan; integration boundary | Done | `docs/vjepa2_ac_architecture.md` |

## 2. Inference timing baseline (V-JEPA 2-AC, bf16, RTX 3090)

One CEM-planned action, 10 iterations, horizon 1. Model load ~18 s; weights 5.0 GiB on GPU.

| CEM samples | whole-batch | `--chunk 200` | predictor (GPU) | peak (chunk) |
|---|---|---|---|---|
| 100 | 4.4 s | 4.4 s | 4.0 s | 8.7 GiB |
| 200 | 8.1 s | 8.1 s | 7.6 s | 9.8 GiB |
| 400 | 15.9 s | 16.1 s | 15.2 s | 11.3 GiB |
| 800 (paper) | 148 s | **32.0 s** | 30.5 s | 15.0 GiB |

Two engineering facts fell out (recorded in `docs/lessons_learned.md` #14-15):

- **bf16 is required.** fp32 disables torch's fused flash / memory-efficient attention
  (math-kernel fallback), so encode + CEM run under `torch.autocast(bf16)`.
- **Chunk the CEM sample batch.** Whole-batch 800 hit a CUDA allocator cliff (148 s wall,
  but the predictor was only 31 s -- ~115 s of synchronous malloc/free thrash above ~12 GiB
  peak). Sub-batching through the predictor (`--chunk`, default 200) keeps peak in the
  linear regime, is numerically identical (each sample is an independent batch row), and
  drops 800 to 32 s. 32 s on the 3090 is consistent with the paper's 16 s on a ~1.8x-faster
  4090, so the timing is reproduced.

## 3. Audit and cleanup (rubber-duck, gpt-5.5 xhigh)

Judge-and-jury audit of the inference harness and logging module. No blockers; all four
findings applied and re-validated (timings and the planned action unchanged).

| # | Severity | Finding | Resolution | Validated |
|---|---|---|---|---|
| 1 | should-fix | Encoder load used `strict=False` and only logged counts, so a wrong checkpoint could run silently on unloaded weights | Keep `strict=False` (matches upstream; RoPE has no persistent buffers here) but raise on any unexpected key or non-RoPE missing key | Yes |
| 2 | should-fix | Vendored-`src` isolation was brittle (repo `src` could win under `python -m`, pytest, or an IDE runner) | Evict the repo root from `sys.path` and delete any imported project `src*` modules before importing the vendored package | Yes |
| 3 | should-fix | Device checks only matched the exact string `"cuda"`, so `--device cuda:N` broke sync / peak-memory / timing | Normalize with `torch.device` + `set_device`; guard on `dev.type == "cuda"` and pass `dev` to the CUDA APIs | Yes |
| 4 | nit | `weights_only=False` was unnecessary for a trusted local checkpoint | Load with `weights_only=True` | Yes |

Cleanup performed alongside the fixes:

- Hoisted a per-call `import time` to module scope; removed an orphaned duplicate line.
- Added `pytest.ini` (`testpaths = tests`, `norecursedirs = third_party ...`) so bare
  `pytest` no longer tries to collect the vendored repo's own test suite.
- Synced the docs: `docs/research_log.md` (#9), `docs/lessons_learned.md` (#14-15),
  `CHANGELOG.md`, and `docs/vjepa2_ac_architecture.md`.
- Verified log hygiene: `logs/`, `checkpoints/`, `third_party/`, and `CHANGELOG.md` are
  gitignored, so only source and docs are committed.

## 4. What this stage does NOT include (the experiments)

These are the first real experiments and are intentionally deferred until the setup above
was validated:

- Closed-loop V-JEPA 2-AC control in MuJoCo (send a frame, plan an action, step the env).
- Interface calibration: the App. B.4 rotation fix and matching the exocentric camera to
  the DROID convention, so zero-shot actions map correctly.
- Zero-shot transfer test on the reach / place tasks in simulation.
- Predictor fine-tuning with a frozen encoder (the plan in `docs/vjepa2_ac_architecture.md`).
