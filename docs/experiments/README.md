# Experiments

Reproducible experiments for the CopilotWorldLab world-model pilot. Each experiment states a
question, method, results (tables + figures under [`results/`](../../results)), and honest
limits. Style follows the I-JEPA_3D_OCT experiment docs.

## Results summary

| Experiment | Question | Headline result | Detail |
|---|---|---|---|
| **Energy-landscape reproduction** | Does the loaded V-JEPA 2-AC reproduce the paper's behavior? | PASS — energy min near the ground-truth action (reverse cos **+0.98**, err 0.030 m); reverse flips | [energy_landscape_and_camera_ablation.md](energy_landscape_and_camera_ablation.md#result-1--paper-example-trajectory-correctness-gate-pass) |
| **Camera-placement ablation** | Which exocentric camera transfers best zero-shot? | az45_el45 best (mean cos **+0.92**); built-in exo_cam worst (-0.16); **improvement +1.08** | [energy_landscape_and_camera_ablation.md](energy_landscape_and_camera_ablation.md#result-2--camera-placement-ablation) |
| **View-relative frame** | Is a weak camera unusable, or just uncalibrated? | Horizontal frame is view-relative; fitted W* rotation tracks azimuth; most side cameras recover to cos **0.70-0.95** | [energy_landscape_and_camera_ablation.md](energy_landscape_and_camera_ablation.md#result-3--the-horizontal-action-frame-is-view-relative-confound-resolved) |
| **Transition scoring (benchmark 1)** | Does vanilla V-JEPA 2-AC understand real transitions? | True action beats random negatives on 300 real DROID transitions — rank **0.820** vs different-episode null **0.486** (+0.334 goal-image effect), top1 0.320 | [transition_scoring.md](transition_scoring.md) |
| **Closed-loop CEM (Phase 1)** | Can V-JEPA 2-AC plan to a goal image in a control loop, and chain sub-goals? | Reach **succeeds** (goal image in 3 steps, 2 cm); 2-goal chain advances sub-goals; lateral sub-goal plateaus at ~3-4 cm (vanilla precision floor) | [cem_closed_loop.md](cem_closed_loop.md) |
| **Closed-loop task success (Phase 1)** | How well does vanilla V-JEPA 2-AC do Reach / Grasp-Lift / Place, and at what precision? | Reach **5/5**; Grasp-Lift **3/5**; Place **0/5** (plateaus ~15 cm vs 6 cm zone) — hidden-state success, 5 trials/task | [closed_loop_benchmark.md](closed_loop_benchmark.md) |

Primary finding: **camera angle is the dominant zero-shot knob** — moving from the built-in
`exo_cam` to an over-the-shoulder az45_el45 view improves action-alignment cosine by +1.08 with
no model change. The best view (`PLANNING_CAMERA` in `src/envs/franka_build.py`) needs almost no
interface calibration.

Secondary finding: the "weak" cameras are **uncalibrated, not unusable** — the horizontal action
frame is view-relative, so a fixed per-camera W* rotation (paper App. B.4) recovers them. Only
the near top-down view has a genuine depth-observability failure.

## Structure

```
docs/experiments/
  README.md                                    this index
  benchmark_plan.md                            established-benchmark strategy + metrics + stack
  transition_scoring.md                        benchmark 1: vanilla V-JEPA 2-AC on real DROID transitions
  energy_landscape_and_camera_ablation.md      correctness gate + camera ablation + frame analysis
  cem_closed_loop.md                           Phase 1 pilot: closed-loop CEM planning to goal image(s)
  closed_loop_success_plan.md                  closed-loop task-success benchmark strategy
  closed_loop_benchmark.md                     closed-loop task success: setup, criteria, logging, results
results/benchmarks/closed_loop_smoke/          Reach/Grasp/Place rollout GIFs + contact sheets + trials CSV
results/benchmarks/
  droid_transition_scoring.png                 rank_frac distribution + image-conditioning control
  droid_transition_scoring.csv                 per-transition source rows
  droid_transition_scoring_table.md            summary metric table
results/camera_ablation/
  camera_grid.png          the 8 camera angles, annotated with cosine / rotation / verdict
  camera_ranking.png       per-camera mean-cosine bar chart
  frame_rotation.png       fitted W* rotation vs camera azimuth (view-relative evidence)
  energy_landscape_*.png   paper-example energy heatmaps (correctness gate)
  combined_table.md        per-camera summary with improvement-over-baseline
  per_camera_tables.md     one per-axis table per camera angle (worst -> best)
```

## Reference

- [../research_log.md](../research_log.md) — chronological problem/solution log + bibliography
- [../lessons_learned.md](../lessons_learned.md) — mistakes, fixes, invariants
- [../vjepa2_ac_architecture.md](../vjepa2_ac_architecture.md) — compute budget + fine-tune plan
- [../../archive/docs/setup_stage.md](../../archive/docs/setup_stage.md) — pre-experiment setup record (archived)
