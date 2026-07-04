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
| **Transition scoring (benchmark 1)** | Does vanilla V-JEPA 2-AC understand transitions? | True action beats random negatives — rank **1.00** vs shuffled-goal null **0.30**, AUROC 0.953 (DROID example) | [benchmark_plan.md](benchmark_plan.md#1-transition-prediction--action-ranking--implemented-and-run) |

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
  energy_landscape_and_camera_ablation.md      correctness gate + camera ablation + frame analysis
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
- [../setup_stage.md](../setup_stage.md) — pre-experiment setup record
