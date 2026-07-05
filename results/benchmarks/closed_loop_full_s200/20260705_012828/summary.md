# Closed-loop benchmark -- run 20260705_012828

Config: model **vjepa2-ac-vitg (ViT-g encoder 1.01B + AC predictor 305M)**, protocol **single_goal**, samples **200**, cem_steps **10**, rollout **T=2**, topk **10**, maxnorm **0.05 m**, dtype **bf16**, **50 trials/task**, seed 0.
Commit `c047c884dc`. Success = error < threshold AND physical gates (lifted/held/upright/stable/released), judged from hidden MuJoCo truth. Cube/target positions randomized per trial.

## Precision curve (success rate at multiple thresholds, one run)

| task | n | mean err (cm) | median | p90 | @5cm | @3cm | @1.5cm |
|---|---|---|---|---|---|---|---|
| **reach** | 50 | 2.5 | 2.4 | 3.5 | 96% | 70% | 24% |

- reach failures: {'too_far': 2}
- reach mean steps 5.7 (V-JEPA 4.7), mean CEM 25.0 s/step

## Task decomposition (what V-JEPA does vs scripted)
- **reach**: pure V-JEPA closed-loop to a goal image.
- **grasp_lift**: V-JEPA reaches the grasp pose; only close+lift scripted (error = object-EE xy before close).
- **place**: scripted grasp, then V-JEPA drives the held cube over the zone; release lowers straight down (error = object-zone xy).

Plots: `<task>_summary.png` (error histogram, precision curve, failure types, error-vs-energy). Selected GIFs/contact sheets: 3 best/median/worst per task. Full per-step logs + config: gitignored `logs/closed_loop_runs/<run_id>/`.
