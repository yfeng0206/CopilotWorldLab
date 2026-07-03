# V-JEPA 2-AC: Architecture, Compute Budget, and Fine-Tuning Plan

Integration reference for hosting Meta's V-JEPA 2-AC (action-conditioned) world model on
a single 24 GB GPU (RTX 3090). Covers the component architecture, exact parameter sizes
measured from the released checkpoint, the memory budget for inference / planning /
fine-tuning, and a plan for fine-tuning the predictor with a frozen encoder as in the
paper. Numbers are measured from `checkpoints/vjepa2-ac-vitg.pt` or cited from
arXiv:2506.09985; verify before relying on them.

## 1. Overview

V-JEPA 2-AC is a two-part latent world model plus a planner:

- Encoder `E` (frozen ViT-g): a 256x256 RGB frame -> latent tokens `z`.
- Action-conditioned predictor `P` (block-causal transformer): `(z_k, s_k, a_k) -> z_{k+1}`
  in latent space, where `s` is the 7-D end-effector state and `a` the 7-D action delta.
- Planner (CEM / MPC): given a goal image `z_g`, minimize the latent energy
  `E(a) = || P(a; z_k, s_k) - z_g ||_1` over sampled action trajectories.

The encoder is trained self-supervised on internet video and kept frozen; only the
predictor is post-trained on a small amount of robot (DROID) interaction data. This
freeze-the-encoder design is what makes a single 24 GB GPU sufficient (Section 5-6).

## 2. Component Architecture

### Encoder (frozen)

| Component | Detail |
|---|---|
| Architecture | ViT-g/16 (`vit_giant_xformers`; uses torch SDPA, xformers not required) |
| Input | 256x256 RGB, tubelet 2x16x16 (a single frame is duplicated to 2 frames) |
| Embed dim | 1408 |
| Patch grid | 16x16 = 256 tokens per frame |
| Positional encoding | 3D-RoPE |
| Output | `z` of shape [tokens, 1408]; normalized with LayerNorm |
| Parameters | 1,012,173,952 (~1.01B), measured |

### Action-Conditioned Predictor (`vit_ac_predictor`)

| Component | Detail |
|---|---|
| Architecture | VisionTransformerPredictorAC, block-causal attention |
| Input dim | 1408 (encoder embed dim) |
| Predictor embed dim | 1024 |
| Depth | 24 blocks |
| Attention heads | 16 |
| Action / state encoders | Linear(7 -> 1024) each (`action_embed_dim = 7`) |
| Positional encoding | 3D-RoPE on patch tokens; temporal RoPE on action/state tokens |
| Forward | `predictor(reps, actions, poses) -> next-frame tokens` (last `tokens_per_frame`) |
| Parameters | 305,220,992 (~305M), measured |

### Action / state representation

7-D vector `[dx, dy, dz, d_roll, d_pitch, d_yaw, d_gripper]` (position, extrinsic-XYZ
Euler, gripper). This matches `FrankaDroidEnv`'s action layout exactly. CEM samples xyz +
gripper and zeros rotation by default; each translation axis is clipped independently to
`[-maxnorm, maxnorm]` with `maxnorm = 0.075` -- an axis-aligned box (L-inf ball), not an
L1 ball, so up to ~13 cm Euclidean displacement per action at 4 fps. Note: `FrankaDroidEnv`
instead bounds the L2 norm of the translation to `max_translation` (0.13 m); reconcile the
box-vs-L2 shapes during interface calibration before zero-shot transfer.

## 3. The Released Checkpoint

`checkpoints/vjepa2-ac-vitg.pt` (11.76 GB, public and ungated on `dl.fbaipublicfiles.com`)
is a training checkpoint, not an inference-only bundle. Top-level keys:

| Key | Content | Params / value |
|---|---|---|
| `encoder` | frozen ViT-g encoder weights | 1.01B (fp32) |
| `predictor` | AC predictor weights | 305M (fp32) |
| `target_encoder` | EMA copy of the encoder (not needed for inference) | 1.01B (fp32) |
| `opt`, `scaler` | AdamW + AMP scaler state (not needed for inference) | - |
| `epoch` / `loss` | 315 / 0.4865 | - |
| `batch_size` / `world_size` / `lr` | 8 / 32 / 4.25e-4 (eff. batch 256) | - |

Inference loads only `encoder` + `predictor` (~1.32B params). The extra ~8 GB on disk is
the EMA encoder plus optimizer state.

## 4. Loading (planned)

The vendored repo's `torch.hub` entry (`vjepa2_ac_vit_giant`) rebuilds `E` and `P` then
downloads weights; two adjustments are needed for our setup:

- The cloned `VJEPA_BASE_URL` is a localhost test stub, so we build with `pretrained=False`
  and load our local checkpoint's `encoder` / `predictor` state dicts directly.
- The vendored repo ships its own top-level `src` package, which collides with ours. The
  loader/inference therefore runs in an isolated module or subprocess that owns the
  vendored `src` namespace (see Section 8), never imported alongside our `src`.

## 5. Memory Budget (single 24 GB GPU)

Weights, per precision (measured param counts; 1 GiB = 2^30 B):

| Tensor | fp32 | bf16 / fp16 |
|---|---|---|
| Encoder (1.01B) | ~3.77 GiB | ~1.89 GiB |
| Predictor (305M) | ~1.14 GiB | ~0.57 GiB |
| Encoder + Predictor | ~4.90 GiB | ~2.45 GiB |

### 5.1 Inference (encode + single rollout)

Frozen, no gradients. Weights (bf16 ~2.45 GiB) + activations for one 256x256 clip and a
short predictor rollout (order ~1 GiB). **Total ~4 GiB. Fits with large headroom.**

### 5.2 Planning (CEM / MPC)

CEM evaluates the predictor over a batch of `samples` action trajectories, so activation
memory scales roughly linearly with `samples` and with the rollout horizon. The paper runs
**800 samples x 10 iterations, horizon 1, on a single RTX 4090 (24 GB), ~16 s/action**.
Our RTX 3090 has the same 24 GB, so the full-fidelity paper setting fits. The notebook's CPU
demo uses only 25 samples x 2 iterations for speed, not memory reasons.

Planning-cost knobs: `samples` (population), `cem_steps` (iterations), `rollout` (horizon).
Memory ~ `samples`; latency ~ `samples x cem_steps x rollout`.

**Measured (RTX 3090, bf16, 10 iters, horizon 1; `scripts/vjepa2_ac_infer_test.py`):**

| samples | whole-batch | chunk=200 | predictor (GPU) | peak (chunk=200) |
|---|---|---|---|---|
| 100 | 4.4 s | 4.4 s | 4.0 s | 8.7 GiB |
| 400 | 15.9 s | 16.1 s | 15.2 s | 11.3 GiB |
| 800 (paper) | 148 s | **32.0 s** | 30.5 s | 15.0 GiB |

Two engineering facts fall out of this (see `docs/lessons_learned.md` #14-15):
- **Use bf16.** fp32 disables torch's fused flash/mem-efficient attention (math-kernel
  fallback), so encode + CEM run under `torch.autocast(bf16)`.
- **Chunk the CEM sample batch through the predictor.** Whole-batch 800 hits a CUDA
  allocator cliff (148 s wall, but predictor only 31 s -- ~115 s of synchronous
  malloc/free thrash above ~12 GiB peak). Sub-batching (`--chunk`, default 200) keeps peak
  in the linear regime and is numerically identical; 800 drops to 32 s. 32 s on a 3090 is
  consistent with the paper's 16 s on a ~1.8x-faster 4090.

### 5.3 Fine-tuning the predictor (frozen encoder)

Only `P` (305M) is trained; `E` (1.01B) is frozen (no grad, no optimizer). AdamW on `P`:

| Component | fp32 |
|---|---|
| Predictor params | ~1.14 GiB |
| Predictor grads | ~1.14 GiB |
| AdamW moments (m, v) | ~2.28 GiB |
| Predictor optimization subtotal | ~4.55 GiB |
| Frozen encoder (bf16, resident) | ~1.89 GiB |
| Activations (small batch; see 6.2) | ~few GiB |
| **Total** | **~8-12 GiB. Fits comfortably.** |

Contrast - full end-to-end (unfreezing `E`, ~1.32B trainable): params + grads + AdamW
moments alone are ~1.32B x 16 B ~= **~19-21 GiB** before activations, which would OOM a
24 GB card. This is why the paper (and we) freeze the encoder.

## 6. What We Can Host

- Only the ViT-g AC model is released (encoder ViT-g + AC predictor). There is no smaller
  released AC checkpoint; the ViT-L / ViT-H encoders are public but have no released AC
  predictor. A smaller AC model would require training our own predictor on a smaller
  encoder.
- On our 24 GB 3090 we can host: (a) inference, (b) full-fidelity CEM planning (~800
  samples, matching the paper's 4090), and (c) fine-tuning the predictor with a frozen
  encoder. All three fit.

## 6.2 Fine-Tuning Plan (predictor only, frozen encoder)

Follows the paper's post-training recipe, scoped to one 24 GB GPU.

1. **Decide zero-shot first.** The paper transfers zero-shot to new labs/robots with no
   fine-tuning. Evaluate our MuJoCo Franka setup zero-shot (after the interface calibration
   in `plan.md`) before committing to fine-tuning; only fine-tune if the sim gap is too
   large.
2. **Data.** Collect MuJoCo Franka trajectories with `FrankaDroidEnv`: 256x256 exocentric
   frames at ~4 fps, 16-frame (4 s) clips, with the 7-D EE state per frame and the 7-D
   action deltas between frames (already our action layout).
3. **Precompute latents.** With `E` frozen, encode all frames once and cache `z` (as the
   OCT project caches frozen-probe features). Train `P` from cached `z`, so `E` need not be
   resident during the predictor step - further reducing memory.
4. **Objective.** Teacher-forcing L1 in latent space (Eq. 2) plus a 2-step rollout L1
   (Eq. 3), exactly as in the paper. Loss is the L1 latent distance to the target `z`.
5. **Optimizer.** AdamW on `P` only; small batch with gradient accumulation (the release
   used batch 8/GPU, eff. batch 256, lr ~4.25e-4). bf16 autocast; encoder in bf16.
6. **Validation.** Track the same latent energy used at planning time, and periodically run
   the CEM reach on held-out goals to confirm the fine-tuned predictor plans better.

## 7. Open Questions

- ~~Exact CEM memory at 800 samples on 24 GB~~ Resolved (Section 5.2): 15.0 GiB peak at
  bf16 with `--chunk 200`, 32 s/action on the 3090.
- Whether zero-shot transfer to the MuJoCo exocentric view is good enough to skip
  fine-tuning (depends on the interface calibration and camera match).
- Whether to run V-JEPA in-process (isolated module) or as a separate local inference
  service (see Section 8).

## 8. Integration Boundary (namespace isolation)

The vendored `facebookresearch/vjepa2` repo has a top-level `src` package that collides
with this project's `src`. V-JEPA loading/inference must therefore be isolated so the two
`src` namespaces never share a process import path. Two options:

- Isolated entry script that puts the vendored repo first on `sys.path` and does not import
  our `src` (used for the first smoke test).
- A small local inference service (separate process) that owns the vendored namespace and
  exposes encode / predict / plan over a thin local interface. This also cleanly answers
  the in-loop-vs-separate question: separate is preferred here, partly to avoid the
  namespace clash and partly to isolate the heavy model from the control loop.
