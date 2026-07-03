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
| Test suite | Model, kinematics, render, and pure-utility tests; render tests skip (not fail) without a GL context | Done (29 passed) | `tests/`, `pytest.ini` |
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

### Second audit (code-auditor, gpt-5.5 xhigh)

A broader cross-cutting audit (P0-P3). No P0, and no blocker to the setup stage. Triaged
real vs. deferred; applied the real, low-risk fixes and re-validated (25 tests pass; the
inference table is unchanged -- the bf16 numbers were already the reported ones).

| Area | Finding | Fix |
|---|---|---|
| Inference default | The bare command ran fp32, contradicting the documented bf16 requirement | Default `--dtype bf16` |
| Goal-image preview | `capture_goal_image` set the gripper command but `mj_forward` does not move the driven fingers, so open vs. closed goals rendered alike | Briefly settle the gripper with physics (live state saved/restored); regression test added |
| Env portability | `DEFAULT_MENAGERIE` was a cwd-relative path, so `FrankaDroidEnv()` failed outside the repo root | Anchor the path to the repo root via `__file__` |
| Doc accuracy | The action bound was documented as an "L1 ball" | It is a per-axis box (L-inf ball); also flagged that `FrankaDroidEnv` bounds the L2 norm, to reconcile at calibration |
| Downloader integrity | The streamed checkpoint had no size check | Verify final size == `Content-Length`; keep the `.part` on truncation |
| Doc drift | `docs/architecture.md` still said V-JEPA was scaffold-only; `docs/plan.md` marked done items as TODO | Updated both to match the working harness |
| Camera default | `MujocoPilotEnv` defaulted to `wrist_cam` vs. the preferred `scene_cam` | Default `scene_cam` |

Deferred to the experiment stage (real observations, not defects): a unified dynamic
Franka + vial + holder scene for end-to-end grasp/insert (that is the first experiment
env), HuggingFace-encoder revision pinning, and tighter dependency pins. Lower-priority
nits (doc wording, older-scaffold action-bound enforcement) are noted in
`docs/lessons_learned.md`.

### Re-audit follow-up (code-auditor, gpt-5.5 xhigh)

A second read-only pass confirmed the fixes above and surfaced remaining items; applied the
real ones (suite now **29 tests pass**, inference table still unchanged):

| Area | Finding | Fix |
|---|---|---|
| 7-D action atomicity | Gripper applied before IK acceptance, and `last_action_ok` ignored orientation | The action is now atomic -- on an unreachable target (position *or* orientation residual over tolerance) neither arm nor gripper move; added `ik_rot_fail_tol`. Reach still 5/5 |
| Remaining "L1-ball" | `docs/related_work.md` still said L1-ball | Corrected to a per-axis box / L-inf |
| Stale plan/handoff | `docs/plan.md` still said "wire via torch.hub / first real inference", listed done work as next, and duplicated a checklist line | Rewrote the next-steps list to current reality; de-duplicated |
| Wrapper guidance | Top docstring still said "load via torch.hub"; the local-loader snippet used a positional arg, but `_make_vjepa2_ac_model` is keyword-only | Point to the local checkpoint; fixed the snippet to keyword args |
| Existing-file integrity | The downloader accepted an existing checkpoint without validation | Verify size against the known `AC_EXPECTED_BYTES` (11,760,743,310) for existing and freshly downloaded files |
| Test coverage | No tests for `latent_energy` / the config loader | Added `tests/test_utils.py` |

Reproducibility pin: the vendored `facebookresearch/vjepa2` is at commit `204698b`
(2026-03-23); the MuJoCo Menagerie is fetched via the sparse clone documented in
`src/envs/franka_build.py`. HuggingFace-encoder revision pinning and full SHA256 hashing
remain deferred (encoders are not yet used; the size check catches truncation/corruption).

## 4. What this stage does NOT include (the experiments)

These are the first real experiments and are intentionally deferred until the setup above
was validated:

- Closed-loop V-JEPA 2-AC control in MuJoCo (send a frame, plan an action, step the env).
- Interface calibration: the App. B.4 rotation fix and matching the exocentric camera to
  the DROID convention, so zero-shot actions map correctly.
- Zero-shot transfer test on the reach / place tasks in simulation.
- Predictor fine-tuning with a frozen encoder (the plan in `docs/vjepa2_ac_architecture.md`).
