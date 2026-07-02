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
