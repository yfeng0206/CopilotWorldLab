# Design

The design and the honest novelty claim for CopilotWorldLab. This is a working research
note: quantitative claims are verified against primary sources (see
[`related_work.md`](related_work.md)) but should be re-checked before publication.

## 0. Project roadmap (phases)

The through-line: reproduce V-JEPA 2-AC as a coarse latent world-model controller, benchmark it
honestly, then improve manipulation by **fusing a third-person view with a first-person (wrist)
view** -- first as a coarse-to-fine handoff, then as a learned cross-view latent, and finally as
a single unified latent. We are currently on **Phase 1**. The technical flow for each phase is in
[`architecture.md`](architecture.md) (flowchart).

- **Phase 0 -- Setup + reproduce (done).** Load V-JEPA 2-AC, set up the environment, and
  verify the world model reproduces the paper: energy-landscape sanity, transition scoring on real
  DROID, and closed-loop CEM to goal images. Build our own honest MuJoCo grasp/place env
  (real physics + hidden success) since robosuite's runtime is Windows-blocked.
- **Phase 1 -- Fixed-bundle closed-loop benchmark (current).** Run the closed-loop task-success
  benchmark on **fixed, saved task bundles** (reproducible, inspectable) over the paper's **four robot
  tasks** (Reach / Grasp / Reach-with-object / Pick-Place; arXiv 2506.09985 Table 3) and **two
  objects** (a rim-graspable **cup** and a rigid **box**) on the same env with the target geom
  swapped, **50 trials per (task, object)** (400 bundles). Success is judged from hidden privileged
  sim state as a Euclidean delta within a swept sphere radius `x` (mean delta + success@x). Ablation
  axes: **CEM population** (200 vs 400 vs 800 samples) and, later, `W*` frame calibration and a
  fine-tuned predictor -- each scored on the identical bundle set. See
  [`experiments/closed_loop_benchmark.md`](experiments/closed_loop_benchmark.md).
- **Phase 2 -- POV/wrist CNN coarse-to-fine (improvement #1).** Add a CNN backbone for the
  first-person **wrist** view. V-JEPA plans the coarse approach (third-person); when it stalls, a
  CNN-based classical image matcher/servo takes over for the fine, close-range motion. Flow:
  `frame -> encoder -> execute (V-JEPA coarse) -> CNN(wrist frame) -> execute (classical fine)`.
  This is the coarse-to-fine handoff (the model's predictive energy gates the switch).
- **Phase 3 -- Third + first view cross-attention (improvement #2, our method).** Third-person
  view -> frozen encoder -> latent end-state (e.g. `[1, 1024]`); a learned **latent transform**
  (e.g. `1024 -> 1024`); then **cross-attention (Q,K,V)** with the wrist/POV view in pixel space
  (patched or raw-pixel); out comes a corrected latent end-state, projected down to an action.
  Flow: `frame -> encoder -> latent -> latent-transform -> cross-attn(POV) -> corrected latent ->
  execute`. Exact depth (number of transform / attention layers) is TBD.
- **Phase 4 -- Unified cross-view latent (final).** Converge the third-person and first-person
  views into a **single latent space** (no pixel-space cross-attention), so both views inform one
  representation and planning improves further. Method is open -- requires a literature search.

Downstream application context (self-driving-lab labware insertion) and the confidence-gate thesis
are unchanged; the phases above develop the coarse controller that the application needs.

### Embodiment: Franka in sim, UR7e on hardware

- **Simulation (now): Franka Panda + Robotiq 2F-85.** We deliberately use a Franka arm in MuJoCo
  because V-JEPA 2-AC was trained on real Franka/DROID video (arXiv:2506.09985) -- matching the
  paper's embodiment is what makes the zero-shot reproduction and any sim benchmark faithful
  ("paper authenticity"). `FrankaDroidEnv` is this substrate.
- **Physical (later): Universal Robots UR7e + Robotiq 2F-85/2F-140.** The real self-driving-lab
  system in the project proposal runs on a UR7e with an Intel RealSense **D405 wrist camera** and
  on-robot Jetson Thor inference. The sim-trained interface/policy transfers to the UR7e for the
  physical demo; the 7-D EE contract is embodiment-agnostic, so only the arm/URDF and calibration
  change, not the world-model interface.
- **Wrist/POV camera = RealSense D405.** The first-person view that Phases 2-4 add corresponds to
  the proposal's D405 wrist camera used for the classical visual-servo seat. Our roadmap extends
  that coarse-to-fine handoff from a classical servo (proposal Layer C) toward CNN matching
  (Phase 2) and a learned cross-view latent (Phases 3-4).

### Relation to the project proposal (staged plan)

This repository is the **Stage-1 simulation** substrate of the broader "Copilot Automation Lab"
proposal (learned world-model manipulation for self-driving chemistry labs; target CVPR/NeurIPS
2027 workshop). The proposal's three control layers -- (A) LLM planner + deterministic scheduler,
(B) world-model coarse arm control, (C) classical visual-servo precise seat, with the world
model's own confidence as the B->C gate -- are described in Sections 2-5 below. The current code
implements **Layer B, Option 1 (V-JEPA 2-AC MPC)**; Option 2 (a VLA-JEPA-style feed-forward
policy), Layers A and C, and the confidence gate are future work. The proposal PDF itself is not
tracked in this repo.

## 1. One paragraph

Self-driving chemistry labs already automate experiment selection, but the robot arm is
still run by classical, pre-programmed control that needs per-station calibration and
custom labware. This project adds two learned layers on top of an otherwise standard
lab: (i) an LLM planner that turns a natural-language request into a schedule of typed
machine actions and re-plans as results arrive, and (ii) a world-model arm controller
that performs the variable coarse approach to and placement of labware. Classical
control is kept for the precise and safety-critical steps. The distinctive thesis: use a
latent world model for the coarse motion, and use that same model's own predictive
confidence as the gate that decides when to hand off to a precise, deterministic seat.

## 2. Control layers

- **A. LLM planner + deterministic scheduler.** The LLM plans at a slow cadence and
  never issues real-time motor commands. A deterministic scheduler holds a backlog,
  resolves resource locks (arm + source + destination acquired atomically, so no
  deadlock), dispatches, and parks the arm when idle.
- **B. World-model coarse arm control.** A latent video world model performs the
  variable coarse approach and placement. Two backends trade flexibility against speed:
  - *Option 1 (primary substrate here): V-JEPA 2-AC with model-predictive control.*
    Plans to a goal image by minimising a latent energy via the Cross-Entropy Method;
    flexible, needs no task-specific demonstrations, but slow (~16 s per action on an
    RTX 4090 in the paper's setup). This is what the MuJoCo pilot is built to exercise.
  - *Option 2: a fast feed-forward latent-world-model policy* (e.g. VLA-JEPA-style) that
    needs a small set of demonstrations; suitable for an on-robot accelerator.
- **C. Classical precise control.** A vision-only image-based visual servo nulls the
  pixel error between the holder opening and the gripper axis to the millimetre scale; a
  passive chamfer completes the seat. No force/tactile sensor.
- **Gate.** The world model's own signal -- predictive energy (Option 1) or ensemble
  disagreement (Option 2) -- is the candidate handoff signal from B to C. Below
  threshold: bounded retry, then flag a human.

The central measured question: **does that self-confidence signal reliably predict a
failed handoff?** Reported as ROC AUC against a simple baseline and against the visual
servo's pixel-error convergence. A negative result is itself informative.

## 3. Data flow (coarse stage, Option 1)

```
goal image  x_g ---------------------------> encode (frozen ViT-g) ---> z_g
current RGB x_k --> encode (frozen ViT-g) --> z_k                         |
end-effector state s_k (7-D) ------------------------------------------+  |
                                                                       v  v
        CEM over action sequences a_1:T  minimising  E = || P(a; s_k, z_k) - z_g ||_1
                                                                       |
                              execute first action, re-plan (receding horizon MPC)
                                                                       |
                              energy magnitude / disagreement  ---> GATE ---> classical seat
```

`P` is the action-conditioned predictor; the energy `E` is the L1 latent distance to the
goal. The MuJoCo env (`src/envs/franka_droid_env.py`, `FrankaDroidEnv`) provides `x_k`
(render), `s_k` (`get_ee_state`), applies `a` (`apply_action` -- real IK + physics), and
produces `x_g` (`capture_goal_image`). See [`architecture.md`](architecture.md) for the exact
signatures. `FrankaDroidEnv` is a real 7-DoF arm with differential IK, contacts and a Robotiq
gripper (grasping is physical), so it exercises real arm dynamics, not a kinematic toy; the
frame, scale, camera and cadence of the action interface must still be calibrated against the
off-the-shelf checkpoint before any planning result is trusted (see the camera ablation in
[`experiments/energy_landscape_and_camera_ablation.md`](experiments/energy_landscape_and_camera_ablation.md)
and [`plan.md`](plan.md)).

## 4. The novelty claim (stated narrowly and honestly)

**Not the novelty.** "One model, one training loop, plan in latent world state to a goal
latent" is the base world-model recipe and comes for free with the backbone. Do not
claim single-model / single-loop as the contribution.

**The novelty (no prior art found for the combination):**
1. A latent world model (not a vision-language-action policy) as the coarse controller.
2. That model's own predictive energy / ensemble disagreement as the handoff gate to a
   classical, deterministic, vision-only visual-servo seat.
3. A viewpoint-conditioned, online-updated goal latent, re-derived from the current
   point of view so fine alignment is calibration-robust, inside the one latent model.
4. Application to vision-only self-driving-lab labware insertion.

**One-line claim.** Coarse-to-fine manipulation carried out with a latent world model by
continuously re-deriving the goal latent from the current viewpoint, with the model's own
predictive energy as the competence gate to a classical vision-only seat, evaluated on
self-driving-lab insertion -- and the reliability of that gate treated as the open
question.

## 5. How this differs from the closest prior art

Full, cited notes are in [`related_work.md`](related_work.md). In short:

- **V-JEPA 2-AC** (arXiv:2506.09985) gives us the substrate: the energy
  `E = ||P(a;s_k,z_k) - z_g||_1` and CEM planning, and reports the energy landscape is
  "smooth and locally convex." It does **not** implement any energy-threshold confidence
  or handoff gate -- that is our addition.
- **AHEAD** (arXiv:2606.02486) computes a predictive-uncertainty signal but spends it on
  truncating the world-model rollout *horizon* inside a frozen OpenVLA loop; it never
  hands off to a classical controller.
- **DreamTacVLA** (arXiv:2512.23864) uses a frozen V-JEPA 2 as a *tactile* encoder and
  refines on a fixed two-pass schedule; it is visuo-tactile, end-to-end, with no
  confidence gate and no classical fallback.
- **VLA-JEPA** (arXiv:2602.10098) uses a JEPA world-model objective only at *training
  time*; the world model is not an inference-time rollout, and the paper uses no
  ensembles or uncertainty. Our confidence gate is a clean novelty boundary.
- **Siamese visual servo** (arXiv:1903.04713) is the load-bearing feasibility citation
  for the vision-only sub-millimetre seat (0.6 mm, 97.5% on a connector, no force sensor).

## 6. Open decision (resolve before finalising the fine-seat story)

Where does the sub-millimetre fine seat happen?

| | (i) Classical seat (doc as written) | (ii) Fully unified latent |
|---|---|---|
| Fine seat | separate classical visual servo | the latent model itself, via viewpoint-updated goal |
| Novelty | weaker (precision credited to the classical module) | stronger, more unified |
| Feasibility | low risk -- 1903.04713 proves vision-only 0.6 mm | high risk -- latent models are centimetre-scale today |

**Recommended: hybrid.** The viewpoint-updated goal latent drives coarse to
few-millimetre inside the model (the novelty plus the energy gate), and the classical
visual servo is a minimal deterministic final seat / safety fallback. The MuJoCo pilot is
designed to inform this decision empirically; it is not resolved yet.
