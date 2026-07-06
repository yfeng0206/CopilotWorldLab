# Closed-loop benchmark -- samples=400 (50 trials/task)

Same config as the s200 stage but with CEM **samples=400** (vs 200). Verifies whether more CEM
search improves the tasks. **pick_place@400 was intentionally skipped** (STOP flag) -- the user
paused the sweep to iterate on pick_place at s200 first (maxnorm ablation).

## Precision curves (vs s200)

| task | n | mean (cm) | success @ thresholds | s200 for comparison |
|---|---|---|---|---|
| **reach** | 50 | 2.7 | @5cm **96%**, @3cm 78%, @1.5cm 6% | s200: 2.5cm, 96% / 70% / 24% |
| **grasp_lift** | 50 | 2.2 | @6cm **58%**, @5cm 58%, @3cm 58%, @2cm 52% | s200: 1.9cm, 54% / 54% / 52% / 36% |
| **pick_place** | -- | skipped (STOP) | -- | s200: 21.6cm, 6% @10cm |

## Reading

- **Reach**: identical to s200 (96% @5cm) -- reach is **sample-saturated**; 200->400 samples does not
  help a task that already solves in ~5 coarse steps.
- **Grasp-Lift**: a modest gain (54%->58% @6cm; @2cm 36%->52%) -- more CEM search tightens the grasp
  positioning slightly, but the single-goal grip still misses ~42% (`missed` x21).
- **Pick-Place**: not run at 400; the s200 result (6% @10cm) is being iterated with the paper-text
  maxnorm=0.075 (see `closed_loop_full_s200_mn075/`).

Per-task reports are in the timestamped subdirectories. The 800-sample stage was not run (would OOM
at whole-batch; a chunked re-run is prepared but deferred pending the pick_place iteration).
