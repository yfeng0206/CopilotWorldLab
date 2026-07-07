# Benchmark progress -- full800_B (run 20260706_184713)

_Snapshot: 2026-07-06T21:52:21. Merged across runs._

- config: samples **800**, cem_steps 10, horizon T=1, maxnorm **0.075**, camera **B_closer**, dtype bf16
- git commit: `a3b288e79f`
- trials completed: **54** / 500

| task / object | n | success@loosest | mean err (cm) | per-threshold success@x |
|---|---|---|---|---|
| grasp/box | 4 | 0% | 4.5 | @0.06=0.00  @0.03=0.00  @0.02=0.00 |
| grasp/cup | 50 | 40% | 6.6 | @0.06=0.40  @0.03=0.02  @0.02=0.02 |
