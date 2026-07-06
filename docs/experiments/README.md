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
| **Closed-loop task success (Phase 1, in rebuild)** | How well does vanilla V-JEPA 2-AC do **grasp / reach_with_object / grasp_and_reach / pick_place** on **cup vs box**, at what precision? | Rebuilding on **fixed, saved task bundles** (4 tasks x cup/box, 50 trials each = 400); success = delta within swept sphere `x`; results pending | [closed_loop_benchmark.md](closed_loop_benchmark.md) |

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
  transition_scoring.md                        benchmark 1: vanilla V-JEPA 2-AC on real DROID transitions (historical)
  energy_landscape_and_camera_ablation.md      correctness gate + camera ablation + frame analysis (historical)
  cem_closed_loop.md                           Phase 1 pilot: closed-loop CEM planning to goal image(s) (historical)
  closed_loop_success_plan.md                  closed-loop task-success benchmark strategy
  closed_loop_benchmark.md                     closed-loop task success: fixed bundles, cup/box, criteria, logging
tasks/                                         fixed, inspectable task bundles (grasp/reach_with_object/grasp_and_reach/pick_place x cup/box)
  <task>/<object>/<id>/                        start.png, goal.png, goal_1/2.png, arrays.npz, model.xml, contact_sheet.png
results/benchmarks/closed_loop_<tag>/          committed per-run reports (summary.md/csv, per task-object figures) once runs complete
```

Note: the earlier committed experiment artifacts under `results/` (benchmark reports, camera-ablation
and energy-landscape figures, transition-scoring CSVs) and the `archive/` scaffolding were removed in
the clean-slate reset and are recoverable from git history. The historical experiment docs above are
kept as prior-result records; their inline figure links point at those cleared files.

## Reference

- [../research_log.md](../research_log.md) — chronological problem/solution log + bibliography
- [../lessons_learned.md](../lessons_learned.md) — mistakes, fixes, invariants
- [../vjepa2_ac_architecture.md](../vjepa2_ac_architecture.md) — compute budget + fine-tune plan
