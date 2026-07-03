# Research Log

Reference log. Each entry records a problem or decision, the investigation, the solution
or decision, and outcomes. New entries are appended at the top of each section.

## Contents
- [Session Log](#session-log) -- chronological problem/decision entries
- [Paper Bibliography](#paper-bibliography) -- every paper referenced, with context
- [Backlog / Future Work](#backlog--future-work) -- planned but not yet done
- [Corrections](#corrections) -- claims we got wrong and the correct answer

---

## Session Log

### 2026-07-02 -- Stage-1 set-up: environment, repo restyle, MuJoCo scaffold

#### 9. First V-JEPA 2-AC inference: logging, timing, and the 800-sample memory cliff
**Context**: With the model verified, load it from the local checkpoint and time one planned
action (encode context + goal, run CEM-MPC on the latent energy) against the paper's ~16 s
on a 4090, using JEPA-style logging. Do a small-to-big sample sweep.
**Setup**: `scripts/vjepa2_ac_infer_test.py` loads ViT-g encoder (strict=False, RoPE) +
AC predictor (strict) from `checkpoints/vjepa2-ac-vitg.pt`, encodes the vendored
`franka_example_traj.npz`, and runs `cem(...)` per config. Logging via `src/utils/logging.py`
(CSV + file + console). Isolated from our own `src` by loading the logging module by file
path before the vendored repo's top-level `src` shadows the package.
**Investigation**:
- fp32 is far slower than bf16: torch SDPA only fuses flash / mem-efficient attention in
  fp16/bf16, so fp32 falls back to the math kernel. All timing below is bf16.
- bf16 sweep (RTX 3090, samples x 10 iters x horizon 1) was linear up to 400 but then
  exploded: 100 -> 4.4 s, 200 -> 8.1 s, 400 -> 15.9 s, **800 -> 148 s**.
- Added a per-call breakdown (predictor GPU time vs pose CPU time). At 800 the predictor was
  only 31 s and the CPU pose update 1.2 s -- so ~115 s was pure framework overhead, appearing
  exactly when peak memory crossed ~12 -> 17 GiB. Classic CUDA allocator thrash: once the
  activation working set is large enough, PyTorch issues synchronous cudaMalloc/cudaFree
  every step. It was never a compute problem.
**Solution**: chunk the CEM sample batch through the predictor (`--chunk`, default 200) so
peak stays in the linear regime. Chunking is numerically identical (each sample is an
independent batch row): the planned action is unchanged at `(+0.075, +0.075, +0.075, gripper
+0.675)`.
**Outcome** (bf16, chunk=200, RTX 3090): 100 -> 4.4 s, 400 -> 16.1 s, **800 -> 32.0 s**
(predictor 30.5 s, peak 15.0 GiB), a 4.6x speed-up at 800 with lower memory. 32 s on a 3090
is consistent with the paper's 16 s on a 4090 (~1.8x faster GPU): timing reproduced. Model
load 18 s, weights 5.0 GiB on GPU. CSV timings in `logs/vjepa2_ac_timing_*.csv`.
**Audit** (rubber-duck, gpt-5.5 xhigh): no blockers; applied all four findings -- fail loudly
on a genuine encoder checkpoint mismatch (keep `strict=False` per upstream but raise on
unexpected keys or non-RoPE missing keys), harden the vendored-`src` isolation (evict repo
root + any project `src*` modules before importing the vendored package), normalize the CUDA
device (`torch.device` + `set_device`, so explicit `cuda:N` works), and load with
`weights_only=True`. Re-validated: identical timings and planned action.
**References**: arXiv:2506.09985 Sec 3 (CEM-MPC, ~16 s/action on 4090);
facebookresearch/vjepa2 notebooks/utils/mpc_utils.py (`cem`, `compute_new_pose`),
src/hub/backbones.py (`_make_vjepa2_ac_model`, `_clean_backbone_key`).

#### 8. V-JEPA 2-AC compute analysis + fine-tuning plan (doc before code)
**Context**: Before wiring inference, quantify param sizes and whether the model + a
predictor fine-tune fit on the 24 GB 3090, and plan the fine-tune like the paper.
**Findings** (measured from `checkpoints/vjepa2-ac-vitg.pt`): encoder 1.012B, predictor
305M (both fp32); file also holds a `target_encoder` EMA copy + optimizer state (why it is
11.76 GB). Training metadata: epoch 315, eff. batch 256, lr 4.25e-4. xformers is NOT needed
(no import in the repo; ViT-g uses torch SDPA). Two integration snags: the cloned repo's
`VJEPA_BASE_URL` is a localhost stub (so load our local checkpoint directly), and the
vendored repo has its own top-level `src` that collides with ours (must isolate).
**Budget (24 GiB)**: inference ~4 GiB (bf16); CEM 800 samples fits (paper used a 4090,
same 24 GB); predictor fine-tune with frozen encoder ~8-12 GiB. Full end-to-end (unfreeze
encoder) ~19-21 GiB just for optimizer/grad/params -> OOM, hence freeze the encoder.
**Outcome**: wrote `docs/vjepa2_ac_architecture.md` (component tables, memory budget,
fine-tune plan, integration boundary), structured like the OCT repo's architecture.md.
**References**: arXiv:2506.09985 Sec 3; facebookresearch/vjepa2 src/hub/backbones.py,
notebooks; checkpoint state_dict.

#### 7. DROID-style Franka + Robotiq reproduction, robosuite eval, scripted reach, audit
**Context**: Reproduce the paper's robot setup (Franka Panda + Robotiq 2F-85, exocentric
camera, 7-D end-effector control) and pick a benchmark suite.
**Investigation / decisions**:
- Composed Franka `panda_nohand` + `robotiq_2f85` via MuJoCo `MjSpec` (mount at the arm
  flange `attachment_site`); added table, floor, light, and a fixed `exo_cam`.
- Added a differential-IK EE-space controller (`src/utils/ik.py`, damped least squares on
  the site Jacobian) so a 7-D EE delta drives the arm; wrapped as `FrankaDroidEnv`.
- Evaluated **robosuite** as the benchmark suite. It fits on paper (OSC = EE control,
  Lift/PickPlace tasks, Robotiq grippers, `mujoco>=3.3.0` so no downgrade), BUT v1.5.2 is
  incompatible with mujoco 3.10: its OSC controller calls `mj_fullM` with the old 2-arg
  signature (3-arg in 3.10). Deferred; kept our own Franka+IK env.
- Built `scripts/scripted_reach_test.py`: a physically-real prescripted reach (IK ->
  data.ctrl -> mj_step, dynamic servos) to 5 targets with a marker; 5/5 reached, viewer +
  headless.
- Ran a rubber-duck audit (gpt-5.5, xhigh). Applied the clean fixes: EE control/state moved
  from the flange to the Robotiq TCP `2f85_pinch` (the flange was ~15.6 cm off), restored
  the elliptic friction cone + impratio after the spec merge, and made `solve_ik` return a
  non-stale final residual. Deferred the bigger fix: make `apply_action` dynamically
  stepped with a measured gripper opening (currently teleports; fine for reach, wrong for
  grasp).
**Outcome**: Franka+Robotiq loads/renders/controls in EE space; scripted reach passes 5/5;
21 tests pass. robosuite set aside on version grounds.
**References**: MuJoCo Menagerie franka_emika_panda + robotiq_2f85; robosuite v1.5.2
setup.py (`mujoco>=3.3.0`); arXiv:2506.09985 Section 3-4; franka-audit findings.

#### 6. Real Franka Panda in MuJoCo: bring-up + timing (no model)
**Context**: First concrete Stage-1 milestone -- get the official Franka into MuJoCo,
confirm it renders/actuates, and check sim timing against the paper's control cadence.
**Investigation**: Sparse-checked-out `google-deepmind/mujoco_menagerie/franka_emika_panda`
(into the gitignored `third_party/`). The model: dt=0.002, nq/nv/nu=9/9/8 (7 arm joints +
2 tendon-coupled fingers; 7 position-servo actuators taking joint-angle targets + 1 gripper
actuator, ctrl 0-255), end-effector = the `hand` body, a `home` keyframe, and NO camera
defined. Wrote `scripts/franka_smoke_test.py`.
**Outcome**: Loads, renders (256x256, saved to gitignored `outputs/franka_home.png`), and
actuates correctly (commanding joint1 -> 0.6 rad moved the hand 32.8 cm). Timing on the
RTX 3090 box: physics ~63k steps/s (~126x real-time), render ~783 fps, and a full
16-frame / 4 s observation clip built in ~57 ms wall-clock. So MuJoCo is negligible next to
the paper's ~16 s/action ViT-g CEM budget -- the model, not the simulator, is the
bottleneck. Cadence maps cleanly: 0.25 s/action = 125 steps, 4 s clip = 2000 steps.
**Next**: The Franka is joint-space position-controlled, but V-JEPA 2-AC emits 7-D EE-space
deltas, so we need a pose->joint layer (differential IK via `mj_jac`, or a mocap target +
weld) as the "Franka behind the 7-D interface" shim.
**References**: MuJoCo Menagerie franka_emika_panda; arXiv:2506.09985 Section 3.1 (cadence).

#### 5. Full re-read of the V-JEPA 2 paper (local PDF, 48 pp) -- calibration recipe found
**Context**: Re-read the paper end-to-end to lock the interface before wiring inference.
**Findings**: (a) CEM uses 800 samples, 10 iterations over the **top-10** (App. B.2) --
our `PlannerConfig(top_k=10)` is already correct. (b) The model trains only on **left
exocentric** (third-person) DROID views -- validates the pilot default camera `scene_cam`,
not `wrist_cam`. (c) App. B.4 gives an **unsupervised calibration recipe**: the inferred
action axis rotates ~linearly with camera angle (systematic ~1.6 cm error on a ~5 cm
delta), removable by least-squares-fitting a 2x2 rotation `W*` from energy-inferred to
executed (dx, dy) -- exactly our interface-calibration step. (d) The energy minimum is only
*near* ground truth (Fig. 9: ~(0,-0.05) vs (0,-0.1)) -- a systematic offset the confidence
gate must tolerate. (e) 'simulation' appears once, only in the Rubinstein CEM citation
title -- confirms no simulator anywhere.
**Outcome**: Updated `related_work.md` and `plan.md` (step 2) with the calibration recipe;
no code change needed (config/planner already match).
**References**: arXiv:2506.09985 Sections 3-4, App. B.2/B.4, Figs. 9/16.

#### 4. V-JEPA 2-AC checkpoint acquired (download only)
**Context**: Need the action-conditioned weights ready for next session without running
inference now.
**Investigation**: The AC model is not on HuggingFace; it ships as a single `.pt` at
`dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt` containing both `encoder` and
`predictor` state dicts (verified from `facebookresearch/vjepa2/src/hub/backbones.py`).
**Solution**: Wrote `scripts/download_checkpoints.py` (resumable streamed download) and
fetched the AC checkpoint into the gitignored `checkpoints/`.
**Outcome**: 10.95 GB `vjepa2-ac-vitg.pt` downloaded, exit 0. Not loaded (no inference
this session).
**References**: arXiv:2506.09985 Section 3; facebookresearch/vjepa2 README + hub loader.

#### 3. MuJoCo scaffold: mocap end-effector instead of a full arm + IK
**Context**: Step 1 needs a scene that renders RGB and exposes the exact interface V-JEPA
2-AC consumes, but a full Franka + IK is more than the pilot needs tonight.
**Investigation**: V-JEPA 2-AC's action is a 7-D end-effector delta (3 pos + 3 extrinsic
Euler + 1 gripper) and its observation is a single RGB image; no joint-space control is
required by the world model. MuJoCo mocap bodies are kinematic and set directly via
`data.mocap_pos/mocap_quat`.
**Solution**: `assets/mujoco/scene.xml` uses a mocap end-effector proxy (with a wrist
camera) over a tabletop with a free-jointed vial and a fixed holder well.
`MujocoPilotEnv` maps the 7-D vector onto mocap pose + a tracked gripper scalar.
**Outcome**: A clean B-stage contract with no arm/IK. 15/15 tests pass, including render
tests (WGL context works on the 3090 desktop). Swapping in a Franka from MuJoCo Menagerie
later is isolated to the scene + a pose->IK shim.
**References**: arXiv:2506.09985 Section 3.1; MuJoCo Python docs (Renderer, mocap).

#### 2. Repo restyled to the I-JEPA_3D_OCT conventions on a new branch
**Context**: Move from a proposal-only repo to an engineering repo with the same doc
discipline as the I-JEPA project.
**Solution**: New branch `world-model-pilot`. README is overview-first with arXiv-linked
references; `docs/` holds DESIGN, architecture, related_work, research_log,
lessons_learned. The proposal document generator, figure generator, figures and `*.docx`
are gitignored (kept locally). `CHANGELOG.md` is a local working doc, also gitignored.
**Outcome**: Committed tree is code + docs only; proposal workflow stays on disk,
untracked.
**References**: yfeng0206/I-JEPA_3D_OCT layout (README, docs/architecture.md,
docs/lessons_learned.md, docs/research_log.md).

#### 1. Local environment bring-up on Windows with no Python
**Context**: Fresh Windows 11 box: RTX 3090 (24 GB), 32 GB RAM, git present, but only the
Microsoft Store Python stub and no conda.
**Investigation**: `nvidia-smi` confirms 24 GB / CUDA UMD 13.3 (the Win32 WMI VRAM field
wrongly reports 4 GB and was ignored). A 3090 is sufficient for ViT-g inference + CEM;
an A100 is only needed later for fine-tuning.
**Solution**: Installed Python 3.11 (winget, user scope), created `.venv`, installed
PyTorch from the CUDA 12.4 wheel index, then MuJoCo + vision/test deps.
**Outcome**: torch 2.6.0+cu124 sees the 3090 (`cuda_available True`); mujoco 3.10.0
imports; all tests pass. Captured in `scripts/setup_env.ps1` and `requirements.txt`.
**References**: download.pytorch.org/whl/cu124; MuJoCo PyPI docs.

---

## Paper Bibliography

- **V-JEPA 2 / V-JEPA 2-AC** -- arXiv:2506.09985 (Meta FAIR, 2025). The coarse-controller
  substrate: latent energy `E = ||P(a;s_k,z_k) - z_g||_1`, CEM planning (800/10/horizon-1,
  ~16 s/action RTX 4090), zero-shot Franka transfer. Does not implement a confidence gate
  (that is ours). Full notes in [`related_work.md`](related_work.md).
- **OpenVLA** -- arXiv:2406.09246 (Kim et al., 2024). 7B DINOv2+SigLIP + Llama-2, 970k
  OXE trajectories, 256-bin discretised 7-DoF, ~6 Hz RTX 4090. The base VLA AHEAD wraps.
- **AHEAD** (*Intercepting the Future*) -- arXiv:2606.02486 (Syed et al., CMU, 2026).
  Sample-variance (S=5) uncertainty gates the prediction horizon inside frozen OpenVLA;
  never hands to a classical controller.
- **DreamTacVLA** (*Learning to Feel the Future*) -- arXiv:2512.23864 (Ye et al.,
  Northwestern, 2025). Frozen V-JEPA 2 as a tactile encoder + trained MLP forecaster;
  fixed two-pass Think-Dream-Act schedule; visuo-tactile; no gate, no classical fallback.
- **VLA-JEPA** -- arXiv:2602.10098 (Sun et al., 2026). JEPA world-model objective at
  training time only; feed-forward flow-matching action head; 100-demo real result; no
  ensembles/uncertainty (the disagreement gate is ours).
- **Siamese visual servo** -- arXiv:1903.04713 (Yu et al., 2019). Vision-only 0.6 mm /
  97.5% connector insertion; feasibility of the classical precise seat.
- **DROID** -- arXiv:2403.12945 (Khazatsky et al., 2024). The 62 h of Franka video the AC
  predictor is trained on.
- **MuJoCo** -- google-deepmind/mujoco. Simulator + headless renderer for the pilot.

Distant lineage (uncertainty-to-defer / model-uncertainty gating / coarse-to-fine):
KnowNo arXiv:2307.01928, Plan2Explore arXiv:2005.05960, MBPO arXiv:1906.08253, PETS
arXiv:1805.12114, coarse-to-fine imitation arXiv:2105.06411.

---

## Backlog / Future Work

- Wire `VJEPA2ACWorldModel` to the downloaded checkpoint via `torch.hub` and run
  encoder-only inference on a rendered frame (first real inference, next session).
- Implement CEM planning (`plan_action`) following
  `facebookresearch/vjepa2/notebooks/utils/mpc_utils.py`; measure latency on the 3090.
- Collect trials (goal-image reaches with synthetic pose perturbations) to fit and
  evaluate the confidence gate (ROC AUC vs a simple baseline and vs pixel-error
  convergence).
- Swap the mocap end-effector for a Franka from MuJoCo Menagerie behind the same 7-D
  interface; add a pose->IK shim.
- Resolve the open fine-seat decision (classical vs unified latent; see DESIGN Section 6).

---

## Corrections

- **Earlier claim that the DreamTacVLA / AHEAD citations were "wrong" was itself wrong.**
  A full-text read confirms both are accurate as characterised: DreamTacVLA does use a
  frozen V-JEPA 2 (as a tactile encoder), and AHEAD does halt on predictive uncertainty
  (of the horizon). Do not re-introduce those "fixes."
- **Title corrections.** AHEAD's published title is *Intercepting the Future...*;
  DreamTacVLA's is *Learning to Feel the Future...*; V-JEPA 2's title uses "Enable"
  ("...Models Enable Understanding..."). Cite the real titles.
- **VLA-JEPA is not a world-model policy at inference.** Its world model is a
  training-time objective only; the action path is a flow-matching head. Do not describe
  it as planning with a world model at test time.
- **AHEAD gates the horizon, not execution, and uses within-model sample variance**, not
  a multi-network ensemble. State both precisely to avoid an overclaim.
