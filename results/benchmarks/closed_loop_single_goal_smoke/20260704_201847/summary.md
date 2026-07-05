# Closed-loop benchmark -- run 20260704_201847

Config: model **vjepa2-ac-vitg (ViT-g encoder 1.01B + AC predictor 305M)**, protocol **single_goal**, samples **200**, cem_steps **10**, rollout **T=2**, topk **10**, maxnorm **0.05 m**, dtype **bf16**, **5 trials/task**, seed 0.
Commit `6fe4c57472`. Success = error < threshold AND physical gates (lifted/held/upright/stable/released), judged from hidden MuJoCo truth. Cube/target positions randomized per trial.

## Precision curve (success rate at multiple thresholds, one run)

| task | n | mean err (cm) | median | p90 | @5cm | @3cm | @1.5cm |
|---|---|---|---|---|---|---|---|
| **reach** | 5 | 2.4 | 2.5 | 3.1 | 100% | 80% | 20% |

- reach mean steps 5.8 (V-JEPA 4.8), mean CEM 24.4 s/step

| task | n | mean err (cm) | median | p90 | @6cm | @5cm | @3cm | @2cm |
|---|---|---|---|---|---|---|---|---|
| **grasp_lift** | 5 | 2.2 | 2.2 | 3.0 | 60% | 60% | 40% | 20% |

- grasp_lift failures: {'missed': 2}
- grasp_lift mean steps 10.0 (V-JEPA 6.0), mean CEM 24.3 s/step

| task | n | mean err (cm) | median | p90 | @10cm | @6cm | @3cm | @1.5cm |
|---|---|---|---|---|---|---|---|---|
| **place** | 5 | 15.6 | 16.1 | 17.7 | 0% | 0% | 0% | 0% |

- place failures: {'outside_zone': 5}
- place mean steps 17.0 (V-JEPA 8.0), mean CEM 24.3 s/step

## Task decomposition (what V-JEPA does vs scripted)
- **reach**: pure V-JEPA closed-loop to a goal image.
- **grasp_lift**: V-JEPA reaches the grasp pose; only close+lift scripted (error = object-EE xy before close).
- **place**: scripted grasp, then V-JEPA drives the held cube over the zone; release lowers straight down (error = object-zone xy).

Plots: `<task>_summary.png` (error histogram, precision curve, failure types, error-vs-energy). Selected GIFs/contact sheets: 3 best/median/worst per task. Full per-step logs + config: gitignored `logs/closed_loop_runs/<run_id>/`.
