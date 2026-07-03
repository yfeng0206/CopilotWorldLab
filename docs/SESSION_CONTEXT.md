# Session context and handoff

> Note (2026-07-03): this is an early handoff note. For the current status and the
> reproducible record of the completed setup, see [`docs/setup_stage.md`](setup_stage.md);
> for the live roadmap see [`docs/plan.md`](plan.md). Sections below that describe the
> repository layout or build steps in terms of the proposal generators are historical --
> those files are now local-only / gitignored.

This document lets a fresh AI coding session pick up this project without the
prior chat history. It captures the design, the novelty audit (with verified
citations), the open design decision, and the pending edits. It is a working
research note: verify quantitative claims against primary sources before
publication.

---

## 1. Project in one paragraph

Self-driving chemistry labs already automate *experiment selection*, but the
robot arm is still run by classical, pre-programmed control that needs
per-station calibration and custom labware. This project adds two learned layers
on top of an otherwise standard lab: (i) an **LLM planner** that turns a
natural-language request into a schedule of typed machine actions and re-plans as
results arrive, and (ii) a **world-model arm controller** that performs the
*variable* coarse approach to and placement of labware at an instrument.
Classical control is kept for the precise and safety-critical steps. The
distinctive thesis: use a **latent world model** for the coarse motion, and use
**that same model's own predictive confidence** as the gate that decides when to
hand off to a precise, deterministic seat.

## 2. Architecture (control layers)

- **A. LLM planner + deterministic scheduler.** The LLM plans at a slow cadence
  and never issues real-time motor commands. A deterministic scheduler holds a
  backlog, resolves resource locks (arm + source + destination acquired
  atomically, so no deadlock), dispatches, and parks the arm when idle.
- **B. World-model coarse arm control.** Two backends, trading flexibility vs
  speed:
  - *Option 1:* a latent video world model with model-predictive control that
    plans to a goal image by minimizing a latent energy (flexible, no
    task-specific demos, but slow — tens of seconds per action).
  - *Option 2:* a fast feed-forward latent-world-model policy that needs a small
    set of demonstrations (suitable for the on-robot accelerator; the demo's
    primary path).
- **C. Classical precise control.** A vision-only image-based visual servo nulls
  the pixel error between the holder opening and the gripper axis to the
  millimeter scale; a passive chamfer completes the seat. No force/tactile
  sensor.
- **Gate.** The world model's own signal — predictive energy (Option 1) or
  ensemble disagreement (Option 2) — is the candidate handoff signal from B to C.
  Below threshold: bounded retry, then flag a human.

The central measured question: **does that self-confidence signal reliably
predict a failed handoff?** Reported as ROC AUC against a simple baseline and
against the visual servo's pixel-error convergence. A negative result is itself
informative.

## 3. The novelty claim (verified)

**What is NOT the novelty.** "One model, one training loop, plan in latent world
state to a goal latent" is the base world-model recipe (a single
action-conditioned predictor with MPC minimizing latent energy to a goal image;
no separate policy). Do **not** claim single-model / single-loop as the
contribution — it comes for free with the backbone. Use it only to separate this
work from two-component approaches (a world-model objective/wrapper **plus** a
separate action head).

**What IS the novelty (no prior art found for the combination):**
1. A **latent world model** (not a vision-language-action policy) as the *coarse*
   controller.
2. **That model's own predictive energy / ensemble disagreement as the handoff
   gate** to a classical, deterministic, vision-only visual-servo seat.
3. A **viewpoint-conditioned, online-updated goal latent**: the goal/end latent
   is re-derived from the current point of view as the view changes, so fine
   alignment is calibration-robust — done inside the one latent model.
4. Application to **vision-only self-driving-lab labware insertion**.

**Defensible one-line claim:** coarse-to-fine manipulation carried out within a
single latent world model by continuously re-deriving the goal latent from the
current viewpoint, with the model's own predictive energy as the competence gate
to a classical vision-only seat, evaluated on self-driving-lab insertion — and
the reliability of that gate treated as the open question.

## 4. Closest prior art (full-text verified — cite honestly, do not overclaim)

- **DreamTacVLA (arXiv:2512.23864).** Confirmed by full-text read: it *does* use
  "a frozen V-JEPA2 world model" (as a tactile feature extractor / "dream" stage)
  and does coarse-alignment → fine-residual insertion (peg-in-hole, USB). But it
  is **tactile**, **end-to-end**, with **no classical fallback and no confidence
  gate**. Our design differs by being vision-only with a gate + classical seat.
- **AHEAD (arXiv:2606.02486).** Confirmed by full-text read: §4.3 "Adaptive
  Horizon Halting" halts the rollout when prediction uncertainty crosses a
  threshold. But it uses that to gate the *prediction horizon* for
  *moving-object anticipation*, stays inside a frozen VLA (OpenVLA) decoder, and
  never hands off to a classical controller. Our gate instead triggers a
  hand-off between a learned coarse stage and a classical precise stage.
- **VLA-JEPA (arXiv:2602.10098).** JEPA-style leakage-free latent pretraining for
  a VLA policy; the basis for the "competitive from ~100 demonstrations" data
  argument. It is a two-component design (world-model objective + action head).
- **Siamese-CNN visual servoing (arXiv:1903.04713).** Verified: 0.6 mm
  translation / 0.4 deg rotation, and 97.5% success on VGA-connector insertion
  *without any force sensing*, explicitly framed as a final refinement after a
  coarse visual servo. This is the load-bearing feasibility citation for the
  vision-only sub-millimeter seat — it checks out exactly.

**Distant lineage (optional to cite, established concepts we build on):**
KnowNo / "Robots That Ask For Help" (arXiv:2307.01928) — uncertainty → defer to
human; Plan2Explore (arXiv:2005.05960) — world-model ensemble disagreement as
epistemic uncertainty; MBPO "When to Trust Your Model" (arXiv:1906.08253) and
PETS (arXiv:1805.12114) — model-uncertainty gating in model-based RL;
Coarse-to-Fine Imitation Learning (arXiv:2105.06411) — the coarse-then-fine
structure. Claim novelty *narrowly* against these — "no one has done
uncertainty-gated handoff" would be false; the specific combination in §3 is what
is new.

## 5. OPEN DECISION (must resolve before finalizing Section 3.3)

Where does the sub-millimeter fine seat happen? Two framings are in tension:

| | (i) Doc as written | (ii) Fully unified |
|---|---|---|
| Fine seat | separate **classical** visual servo | the **latent model** itself, via viewpoint-updated goal |
| Novelty | weaker (precision credited to the classical module) | stronger, more unified |
| Feasibility risk | low — 1903.04713 proves vision-only 0.6 mm | high — latent models are centimeter-scale today |

**Recommended: hybrid.** The viewpoint-updated goal latent drives coarse →
few-millimeter *inside* the model (the novelty + the energy gate), and the
classical visual servo is a minimal deterministic final seat / safety fallback.
Keeps the feasibility crutch and the unified-latent story. Pick one and rewrite
Section 3.3 + the Section 2 novelty paragraph to match.

## 6. Pending edits (not yet applied to build_proposal.py)

1. **Reference title fix:** the reference currently labelled as test-tube/powder
   manipulation (arXiv:2603.01110) has the real title *"Compact Task-Aligned
   Imitation Learning for Laboratory Automation."* Correct the REFS string.
2. **Sharpen the Section 2 novelty paragraph** to the narrow claim in §3 above,
   contrasting explicitly against DreamTacVLA and AHEAD as characterized in §4.
3. **Resolve §5 (the open decision)** and align Section 3.3 wording.

The document currently builds with 19 references. Earlier characterizations that
called the DreamTacVLA / AHEAD citations "wrong" were themselves mistaken (they
were judged from abstracts); a full-text read confirmed the citations are
accurate. Do not re-introduce those "fixes."

## 7. Methodology note (important)

Verify every citation against **primary sources** — the arXiv API
(`export.arxiv.org/api/query`) and Semantic Scholar API — and read the actual PDF
(not just the abstract) before characterizing a paper's method. Generic
LLM-summary web search tools were observed to hallucinate paper titles and
mischaracterize methods; treat them as unreliable for citations.

## 8. Repo layout and build

- `build_proposal.py` — generates the proposal document. Paths are relative to
  the script directory.
- `make_figures.py` — generates the three schematic figures.
- `fig_concept.png`, `fig_arch.png`, `fig_handoff.png` — generated figures.

```bash
pip install python-docx matplotlib
python make_figures.py
python build_proposal.py
```

The generated `.docx` and any internal working notes are intentionally
git-ignored.
