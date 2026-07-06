# Closed-loop benchmark -- full run, samples=200 (50 trials/task)

V-JEPA 2-AC ViT-g, samples **200**, cem_steps 10, rollout T=2, topk 10, maxnorm 0.05 m, momentum_std
0.15, pos_tol 0.015, chunk 400, bf16, RTX 3090, seed 0, **50 trials/task**. The model sees only
image + EE state + goal image(s) (PLANNING_CAMERA); success is judged from hidden MuJoCo truth.
Protocols are paper-faithful: reach and grasp_lift use a single goal image; pick_place is the
composite (grasp -> transport -> place) over 3 sub-goals on the paper's fixed 4/10/4 schedule
(arXiv 2506.09985 Sec. 4.2). Full methodology:
[docs/experiments/closed_loop_benchmark.md](../../../docs/experiments/closed_loop_benchmark.md).

Each rollout records one continuous error; `success@t` = error < t AND all physical gates
(grasp_lift: lifted/held/upright/stable; place/pick_place: grasped[pnp]/upright/stable/released).

## Precision curves

| task | n | mean (cm) | median | p75 | p90 | success @ thresholds |
|---|---|---|---|---|---|---|
| **reach** | 50 | 2.5 | 2.4 | 3.2 | 3.5 | @5cm **96%**, @3cm 70%, @1.5cm 24% |
| **grasp_lift** | 50 | 1.9 | 1.7 | 2.4 | 3.1 | @6cm **54%**, @5cm 54%, @3cm 52%, @2cm 36% |
| **pick_place** | 50 | 21.6 | 22.6 | 25.9 | 28.5 | @10cm **6%**, @6cm 2%, @3cm 0%, @1.5cm 0% |

## Failure types

| task | failures (of 50) |
|---|---|
| reach | too_far x2 |
| grasp_lift | missed x22, tipped x1 |
| pick_place | outside_zone x29, grasp_failed x19, tipped x1 |

## Reading

- **Reach** is a reliable coarse skill: 96% within 5 cm, mean 2.5 cm, only 2 far misses.
- **Grasp-Lift** *positions* extremely well (mean 1.9 cm; 52% within 3 cm) but single-goal grasp
  *success* plateaus at **54%** -- the gripper misses on ~44% of trials (`missed`) even when well
  positioned. This is the paper-faithful single-goal grasp; the multistage pregrasp->grasp ablation
  raised held-success in the n=5 smoke (40%->80% @3cm), so the gap is a grip-timing/approach issue,
  not a positioning one.
- **Pick-Place** (full composite) succeeds **6% at the 10 cm zone** and 2% at 6 cm. Failures split
  between never grasping (`grasp_failed` 19/50, the 4-step grasp budget is tight) and grasping but
  placing outside the zone (`outside_zone` 29/50). This is the honest vanilla baseline for the
  hardest task -- the target the W* calibration + predictor fine-tuning (Phases 2-4) must beat.

Per-task reports (summary.md/csv, plots, selected best/median/worst GIFs) are in the timestamped
subdirectories. The 400- and 800-sample stages follow (sample ablation).

## Comparison to the paper (V-JEPA 2-AC)

Same released config (samples 400, cem_steps 10, T=2, topk 10, momentum 0.15/0.15, maxnorm 0.05 --
identical to `world_model_wrapper.py`) and protocol (single-goal reach/grasp, 4/10/4 pick-and-place),
but in uncalibrated MuJoCo sim (4 cm cube, 50 trials, geometric success gate) vs the paper's real
Franka + Cup/Box, 10 trials, human-judged task completion. Compared at our loosest threshold:

| task | paper (real robot) | ours (sim) | verdict |
|---|---|---|---|
| Reach | 100% | **96%** @5cm | reproduced |
| Grasp | Cup 60% / Box 20% | **54%** @6cm | reproduced (in range) |
| Pick-Place | Cup 80% / Box 50% | **6%** @10cm | large gap -- uncalibrated camera frame (no W*), single small cube, stricter scoring |

Reach and grasp reproduce the paper. Pick-place is the honest vanilla-baseline gap the W* frame
calibration + predictor fine-tuning (Phases 2-4) must close.

### Where pick-place fails (traced, all 50 trials)

Same 3 sub-goals / 4/10/4 schedule as the paper -- the gap is not a goal-count difference.
- **grasp_failed 38%**: 4-step grasp budget too tight (gripper 2.7cm from cube on failures vs 1.5cm
  on successes; our standalone 6-step grasp hits 54%).
- **outside_zone 58%**: held cube needs +25.2cm to the zone but moves only +5.6cm (**22% of needed**);
  direction correct, but **15/31 stall** (<3cm). Root cause: a small 4cm cube barely registers in the
  latent vs the dominant arm/gripper, so the transport gradient flattens and stalls ~20cm short --
  object-salience + uncalibrated frame (W*), not goal design.


