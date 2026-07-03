# Related Work

Full-text-verified notes on the papers this project builds on and must be honest about.
Every claim carries an arXiv id and a section/table reference; each entry ends with a
one-line statement of how our design differs. Where a paper does *not* support something
we might want to claim, that is flagged explicitly.

Method: every citation was checked against primary sources (arXiv full text, not
abstracts; official repos and model cards). Generic LLM-summary web search was treated as
unreliable.

---

## V-JEPA 2 and V-JEPA 2-AC (arXiv:2506.09985, Meta FAIR, 2025)

The substrate for our coarse controller.

- **Pretraining (encoder).** A latent mask-denoising JEPA objective predicts EMA-teacher
  *representations* of masked spatiotemporal patches with an L1 loss (never pixels;
  Section 2.1, Eq. 1). Tubelet patchification 2x16x16, multi-block masking, 3D-RoPE.
  Sizes: ViT-L 300M, ViT-H 600M, ViT-g 1B (Table 4). Data: VideoMix22M, 22M samples,
  >1M video-hours (Table 1). Pretrain up to 64 frames at 384x384.
- **V-JEPA 2-AC (action-conditioned).** Freeze the encoder; train a ~300M block-causal
  predictor (24 layers, 16 heads, 1024 hidden) on ~62 h of unlabeled DROID Franka video
  (Section 3.1). Per-frame features are 16x16x1408. Action = 7-D end-effector delta
  (3 position + 3 extrinsic Euler + 1 gripper). Trained with teacher-forcing + a 2-step
  rollout loss.
- **Planning.** Given a single goal image encoded to `z_g`, minimise the latent energy
  `E = ||P(a; s_k, z_k) - z_g||_1` (Eq. 5) with the Cross-Entropy Method (population 800,
  10 iterations refining the top-10 samples, horizon 1; App. B.2), execute the first
  action, re-plan (receding horizon). ~16 s/action on one RTX 4090 (Table 3). Action
  sampling constrained to an L1-ball of radius ~0.075 (~13 cm). The energy landscape is
  reported "smooth and locally convex" (Fig. 9), though its minimum is only *near* the
  ground-truth action, not exactly on it (min ~(0,-0.05) vs truth (0,-0.1)).
- **Zero-shot transfer.** Deployed zero-shot on Franka arms in two labs not in DROID,
  with an uncalibrated monocular RGB camera, no task-specific training and no reward
  (Section 4.1). Success (Table 2, avg of two labs): Reach 100%, Grasp Cup 65%, Grasp Box
  25%, Reach-with-Object 75%, Pick-and-Place Cup 80%, Box 65%. (Lab-2-only numbers are
  lower, e.g. Grasp Cup 60%.)
- **Stated limitations.** Camera/coordinate-frame sensitivity (must infer the action axis
  from monocular RGB); long-horizon autoregressive drift; dependence on image goals;
  precision is the gating factor for grasps; 16 s/action latency; monocular RGB only.
- **Camera sensitivity + a calibration recipe (App. B.4).** The model is trained only on
  *left exocentric* (third-person) DROID views and must infer the action coordinate axis
  from the monocular RGB image; the inferred axis rotates roughly linearly with camera
  angle (Fig. 16) -- a systematic ~1.6 cm error on a ~5 cm delta. Because the error map
  `W*` is essentially a pure rotation (condition number ~1.5), it can be removed by an
  unsupervised calibration: drive random actions, least-squares-fit the 2x2 map from
  energy-inferred to executed (dx, dy), and rotate inferred actions by `W*` before use.
  The authors describe but do not apply this. It is directly our interface-calibration
  step, and it argues for a third-person camera rather than a wrist camera.
- **Flag (decisive for us).** The paper does **not** implement any energy-threshold
  confidence or handoff gate. Using the energy magnitude as a competence gate is our
  addition; the smooth/convex landscape is encouraging but not evidence for the gate, and
  the energy minimum carries a systematic offset, so the gate must be robust to that.

**How ours differs.** We reuse this exact energy and CEM planner for coarse motion, then
add what the paper lacks: the energy as a competence gate that hands off to a classical
vision-only visual-servo seat.

---

## OpenVLA (arXiv:2406.09246, Kim et al., 2024)

The open base VLA that much of the related work wraps.

- **Architecture.** 7B parameters: a 600M dual vision encoder (DINOv2 + SigLIP,
  channel-concatenated) + 2-layer MLP projector + Llama-2 7B; outputs 7-D control
  (Section 3.1).
- **Data.** 970k trajectories curated from Open X-Embodiment (single-arm,
  >=1 third-person camera), rebalanced with Octo weights (Section 3.3).
- **Action representation.** 7-DoF, each dimension discretised into 256 bins over the
  1st-99th percentile, overwriting the 256 least-used Llama tokens; next-token
  prediction on action tokens only (Section 3.2). The exact (6-DoF delta pose + gripper)
  split is the standard OpenX convention, inferred rather than verbatim.
- **Cost / limits.** ~6 Hz on an RTX 4090, 15 GB bf16 (Section 3.5); LoRA matches full
  fine-tuning training only 1.4% of parameters (Section 5.3); int4 (7 GB) matches bf16
  accuracy (Table 2). Single-image observations only; base success typically <90%; 6 Hz
  flagged as too slow for high-frequency control (Section 6).

**How ours differs.** We do not use a discretised-token VLA policy as the controller at
all; the coarse controller is a latent world model, and the precise seat is classical.

---

## AHEAD (arXiv:2606.02486, Syed et al., CMU, 2026)

Real title: *Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA
Manipulation.* Our closest "uncertainty-gated world model" neighbour.

- **Method.** A ~4.9M-parameter conditional flow-matching latent world model in a
  **frozen** 7B OpenVLA's feature space, predicting future patch tokens that feed the
  unchanged frozen action decoder (Section 4.2).
- **Uncertainty and halting.** The uncertainty signal is the **sample variance across
  S=5 flow-matching samples** (Eq. 4), not a multi-network ensemble and not entropy. The
  rollout halts at the first horizon step where mean-token variance exceeds `tau_u` (set
  to the 90th percentile of training uncertainty), else at `K_max=10` (Section 4.3).
- **What is gated.** The **prediction horizon** `K` -- how far ahead to imagine -- not
  action execution. Output always returns to the frozen neural decoder.
- **Results.** Simulation 79-97% vs 31-58% for the strongest baseline; a physical xArm 7
  intercepts a projectile launched from 2 m (19/30) where baselines score 0/30.

**How ours differs.** AHEAD spends its uncertainty on truncating the world-model rollout
horizon *inside* the neural loop and never leaves the neural decoder; ours uses the world
model's own signal to hand off control to a classical vision-only visual-servo seat --
a coarse-to-precise controller handoff, not a horizon truncation. (Also: our "ensemble
disagreement" is a different uncertainty construction than AHEAD's within-model sample
variance.)

---

## DreamTacVLA (arXiv:2512.23864, Ye et al., Northwestern, 2025)

Real title: *Learning to Feel the Future: DreamTacVLA for Contact-Rich Manipulation.* The
other "V-JEPA 2 + coarse-to-fine insertion" neighbour.

- **How V-JEPA 2 is used.** As a **frozen tactile feature extractor** (V-JEPA2 ViT-L/g
  on GelSight tactile images), Section 3.1.3, plus a lightweight residual adapter (5.5M
  params, 1.8% overhead). The "dream" future forecast is a **separate trained MLP**
  `F_eta`, not V-JEPA 2's own predictor.
- **Pipeline.** A Think-Dream-Act loop that runs every step on a **fixed two-pass
  schedule**: draft an action, dream the future tactile latent, refine. The coarse ->
  fine transition is an unconditional schedule, learned end-to-end -- there is **no
  uncertainty/confidence gate**.
- **Sensing / fallback.** Visuo-tactile (GelSight + two RealSense cameras + language); no
  classical deterministic seat or fallback anywhere.
- **Tasks / results.** Peg-in-Hole 95.0%, USB Insertion 85.7% ("sub-millimetre,
  ambiguous from vision alone"), Gear Assembly 81.1%, Tool Stabilization 74.6% (Table 1).
  7-DoF (6-D EE pose + gripper). No explicit numeric clearances beyond "sub-millimetre".

**How ours differs.** DreamTacVLA is tactile-centric, end-to-end, refines on a fixed
schedule with no gate and no classical fallback; ours is vision-only and uses the latent
world model's own predictive energy / disagreement as an explicit gate that hands off to
a classical vision-only visual-servo seat.

---

## VLA-JEPA (arXiv:2602.10098, Sun et al., 2026)

The basis for the "competitive from few demonstrations" data-efficiency argument, and the
substrate for the fast feed-forward Option 2.

- **Architecture.** Two-component: a Qwen3-VL-2B backbone carrying a JEPA latent
  world-model pretraining objective, plus a separate DiT-B flow-matching action head
  (Section 3.1). Joint objective `L = L_FM + beta * L_WM` (Eq. 9).
- **Flag (critical for framing).** The V-JEPA2 world model is **training-time only** -- it
  supplies latent supervision targets and is **not** in the inference-time action path.
  At test time the action comes from the backbone -> latent action tokens -> flow-matching
  head (one VLM forward + a 4-step flow-matching solve). Calling it a "world-model policy"
  is only true in the pretraining sense.
- **Data efficiency.** Two distinct results, not to be merged: real-world fine-tuning
  with **100 demonstrations across 3 tasks** (Section 4.1/4.4); and a SimplerEnv result
  using "less than 1% of the training data used by villa-X" (Section 4.3). LIBERO avg
  97.2 (Table 1).
- **Flags.** No latency/throughput/Hz numbers anywhere -- we cannot cite it as "fast
  feed-forward" or "real-time". No ensembles / uncertainty / confidence anywhere -- the
  disagreement gate is entirely our addition. Its real-robot trajectories "rarely breach
  the robot arm's safety constraints" vs pi-0.5 (Section 4.4).

**How ours differs.** For Option 2 we would add, on top of a VLA-JEPA-style policy, an
ensemble-disagreement confidence gate and a classical handoff -- neither of which exists
in the paper.

---

## Vision-only sub-millimetre seat (arXiv:1903.04713, Yu et al., 2019)

The load-bearing feasibility citation for the classical precise stage: a Siamese-CNN
image-based visual servo inserts a VGA connector at ~0.6 mm translation / 0.4 deg with a
97.5% success rate, from vision alone, no force sensing -- explicitly framed as a final
refinement after a coarse visual servo. This is why the world model is not asked for
sub-millimetre precision.

---

## Distant lineage (build on, do not overclaim against)

Uncertainty-to-defer (KnowNo, arXiv:2307.01928); world-model ensemble disagreement as
epistemic uncertainty (Plan2Explore, arXiv:2005.05960); model-uncertainty gating in
model-based RL (MBPO arXiv:1906.08253, PETS arXiv:1805.12114); coarse-to-fine imitation
(arXiv:2105.06411). "No one has done uncertainty-gated handoff" would be false; the
specific combination in [`DESIGN.md`](DESIGN.md) Section 4 is what is new.
