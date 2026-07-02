# CopilotWorldLab

Proposal and figure-generation source for **Learned World Model Manipulation for
Self-Driving Chemistry Laboratories**.

## Idea

Self-driving ("autonomous") chemistry labs already automate experiment selection,
but the robot arm is still run by classical, pre-programmed control that needs
per-station calibration and custom labware. This project adds two learned layers
on top of an otherwise standard lab:

1. **An LLM planner (Copilot)** that turns a natural-language request into a
   schedule of typed machine actions, and re-plans as results arrive.
2. **A world-model arm controller** that performs the *variable* part of arm
   motion — the coarse approach to and placement of labware at an instrument.

Classical control is kept for the precise and safety-critical steps. A latent
world model drives the coarse approach; a vision-only visual servo performs the
sub-millimeter seat; and the world model's own predictive confidence is proposed
as the gate that decides when to hand off from the learned coarse stage to the
precise stage. Whether that confidence signal reliably predicts a failed handoff
is treated as the project's central open question.

## Contents

- `build_proposal.py` — generates the proposal document.
- `make_figures.py` — generates the schematic figures.
- `fig_concept.png`, `fig_arch.png`, `fig_handoff.png` — generated figures.

## Build

```bash
pip install python-docx matplotlib
python make_figures.py      # regenerate figures
python build_proposal.py    # generate the proposal document
```

## Status

Research-stage proposal. References and quantitative claims are a work in progress
and should be verified against primary sources before publication.
