# Transition Scoring — Vanilla V-JEPA 2-AC on Real DROID (Benchmark 1)

**Question.** Does the vanilla (unmodified) V-JEPA 2-AC model actually *understand real robot
transitions* — i.e., does it score the executed action lower in latent energy than random
alternatives, and does that ranking depend on the goal image? This is the first honest,
established-benchmark baseline in the project's evaluation plan
([benchmark_plan.md](benchmark_plan.md)); the fine-tuned predictor will be measured against the
same numbers.

**Why DROID.** The intended established grasp/place source was robomimic (Lift/Can/Square), but
robomimic does not host pre-rendered image datasets — only `low_dim` (no images) and `raw` (sim
states needing a robosuite render, which is blocked on Windows, lessons_learned #11/#18/#19). The
Windows-runnable established alternative is **DROID itself**, the real-robot dataset V-JEPA 2-AC
was trained and evaluated on (arXiv:2506.09985; DROID arXiv:2403.12945). This is the fairest
"does it understand real transitions?" test rather than a self-made easy dataset.

## Method

- **Data.** `lerobot/droid_100` (100 real teleoperated Franka episodes, LeRobot v3.0: parquet
  `observation.state`/`action` + av1 mp4 cameras). We use the `exterior_image_1_left` camera.
  [`scripts/extract_droid_transitions.py`](../../scripts/extract_droid_transitions.py) decodes
  the video, reads per-frame 7-D EE state `[x, y, z, roll, pitch, yaw, gripper]`, and emits one
  npz per transition: `(image_t, state_t) -> (image_{t+H}, state_{t+H})`, `H = 5` frames (~0.33 s
  at 15 fps). Transitions with < 2 cm xyz motion are dropped. **n = 300** transitions from the
  first 20 episodes (15 per episode).
- **Score.** For each transition the true action's xyz translation (recovered from the state
  delta, rotation/gripper zeroed) is compared against **K = 32** random negative directions of
  the same magnitude. Latent energy `E(a) = mean(|P(a; z_ctx, s_ctx) - z_goal|)`. Primary metric
  is the within-transition `rank_frac` (fraction of negatives with higher energy than the true
  action; chance 0.5). [`scripts/benchmark_transition_scoring.py`](../../scripts/benchmark_transition_scoring.py).
- **Null control (image-conditioning).** The identical test is rescored with the goal latent
  taken from a **different episode** (a true scene mismatch). If the model is image-goal-
  conditioned, `rank_frac >> null ~ 0.5`. Using a same-episode neighbour as the null inflates it
  (0.74) because adjacent frames are near-identical; drawing from a different episode is the
  honest control.
- bf16, RTX 3090, seed 0.

## Result

![DROID transition scoring](../../results/benchmarks/droid_transition_scoring.png)

| metric | value | meaning |
| --- | --- | --- |
| transitions (n) | 300 | real DROID `(image_t, state_t) -> (image_{t+H})` pairs |
| **rank_frac** | **0.820** | mean fraction of 32 random negatives the true xyz action beats (chance 0.5) |
| **null rank_frac** | **0.486** | same, goal from a different episode (image-conditioning control) |
| **conditioning gap** | **+0.334** | rank_frac − null; the goal-image effect |
| top1_acc | 0.320 | fraction of transitions where the true action beats ALL 32 negatives |
| gap_z | +1.45 | mean (neg energy − true energy) / std, effect size |
| AUROC (pooled) | 0.612 | true-vs-negative separability pooled across scenes (mixes global calibration) |

Source CSV: [`results/benchmarks/droid_transition_scoring.csv`](../../results/benchmarks/droid_transition_scoring.csv).

## Reading

- **Supported:** vanilla V-JEPA 2-AC is genuinely image-goal-conditioned on real robot data. The
  true executed action is favored over random negatives (0.820) far above the different-episode
  null (0.486); the +0.334 gap cannot come from a fixed action prior, since the prior would score
  identically regardless of goal. The left panel shows most transitions piling up near
  `rank_frac = 1.0`.
- **Honest limits:** this is a *one-step scoring* benchmark, not closed-loop planning success.
  It is harder than the curated paper example (rank 1.00, n=2) because DROID's exterior camera is
  randomized per scene and the 5-frame horizon is short; `top1 = 0.320` and pooled `AUROC = 0.612`
  reflect that per-scene energy is not globally calibrated. Actions are xyz-translation only
  (rotation/gripper zeroed). The absolute number depends on `H`, `min_motion`, camera choice, and
  K; those are fixed and reported so the fine-tuned delta is measured on the same protocol.

## What improvement will look like

The fine-tuned predictor (frozen encoder) is expected to raise `rank_frac`, `top1_acc`, and the
conditioning gap on this **exact** protocol (same n, H, K, seed, camera), and to improve pooled
AUROC via better cross-scene energy calibration. Any such delta is a falsifiable, established-
benchmark improvement rather than a self-made-dataset win.

## Reproduce

```
python scripts/extract_droid_transitions.py --max-episodes 20 --per-episode 15
python scripts/benchmark_transition_scoring.py --traj "outputs/droid_transitions/*.npz"
python scripts/plot_transition_benchmark.py --title "V-JEPA 2-AC transition scoring -- DROID (n=300)"
```

## References

- V-JEPA 2 / V-JEPA 2-AC — arXiv:2506.09985
- DROID — arXiv:2403.12945; dataset `lerobot/droid_100`
- [benchmark_plan.md](benchmark_plan.md) — the full evaluation strategy and metric table
- [../lessons_learned.md](../lessons_learned.md) — #11/#18/#19 (why robomimic/robosuite/ManiSkill
  images are blocked on Windows and DROID is used instead)
