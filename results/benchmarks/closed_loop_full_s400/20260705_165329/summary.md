# Closed-loop benchmark -- run 20260705_165329

Config: model **vjepa2-ac-vitg (ViT-g encoder 1.01B + AC predictor 305M)**, protocol **single_goal**, samples **400**, cem_steps **10**, rollout **T=2**, topk **10**, maxnorm **0.05 m**, dtype **bf16**, **50 trials/task**, seed 0.
Commit `1cbee754d9`. Success = error < threshold AND physical gates (lifted/held/upright/stable/released), judged from hidden MuJoCo truth. Cube/target positions randomized per trial.

## Precision curve (success rate at multiple thresholds, one run)

| task | n | mean err (cm) | median | p90 | @6cm | @5cm | @3cm | @2cm |
|---|---|---|---|---|---|---|---|---|
| **grasp_lift** | 50 | 2.2 | 1.7 | 3.5 | 58% | 58% | 58% | 52% |

- grasp_lift failures: {'missed': 21}
- grasp_lift mean steps 9.8 (V-JEPA 5.8), mean CEM 76.3 s/step

## Task decomposition (what V-JEPA does vs scripted)
- **reach**: pure V-JEPA closed-loop to a goal image.
- **grasp_lift**: V-JEPA reaches the grasp pose; only close+lift scripted (error = object-EE xy before close).
- **place**: scripted grasp, then V-JEPA drives the held cube over the zone; release lowers straight down (error = object-zone xy).

Plots: `<task>_summary.png` (error histogram, precision curve, failure types, error-vs-energy). Selected GIFs/contact sheets: 3 best/median/worst per task. Full per-step logs + config: gitignored `logs/closed_loop_runs/<run_id>/`.
