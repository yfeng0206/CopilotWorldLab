# Closed-loop benchmark -- run 20260705_113458

Config: model **vjepa2-ac-vitg (ViT-g encoder 1.01B + AC predictor 305M)**, protocol **single_goal**, samples **400**, cem_steps **10**, rollout **T=2**, topk **10**, maxnorm **0.05 m**, dtype **bf16**, **50 trials/task**, seed 0.
Commit `179a7974a5`. Success = error < threshold AND physical gates (lifted/held/upright/stable/released), judged from hidden MuJoCo truth. Cube/target positions randomized per trial.

## Precision curve (success rate at multiple thresholds, one run)

| task | n | mean err (cm) | median | p90 | @5cm | @3cm | @1.5cm |
|---|---|---|---|---|---|---|---|
| **reach** | 50 | 2.7 | 2.6 | 4.0 | 96% | 78% | 6% |

- reach failures: {'too_far': 2}
- reach mean steps 5.9 (V-JEPA 4.9), mean CEM 76.9 s/step

## Task decomposition (what V-JEPA does vs scripted)
- **reach**: pure V-JEPA closed-loop to a goal image.
- **grasp_lift**: V-JEPA reaches the grasp pose; only close+lift scripted (error = object-EE xy before close).
- **place**: scripted grasp, then V-JEPA drives the held cube over the zone; release lowers straight down (error = object-zone xy).

Plots: `<task>_summary.png` (error histogram, precision curve, failure types, error-vs-energy). Selected GIFs/contact sheets: 3 best/median/worst per task. Full per-step logs + config: gitignored `logs/closed_loop_runs/<run_id>/`.
