# Benchmark progress -- full800_B (run 20260706_184713, 20260707_010843, 20260707_035033)

_Snapshot: 2026-07-07T08:16:06. Merged across runs._

- config: samples **800**, cem_steps 10, horizon T=1, maxnorm **0.075**, camera **B_closer**, dtype bf16
- git commit: `a3b288e79f`
- trials completed: **150** / 500

| task / object | n | success@loosest | mean err (cm) | per-threshold success@x |
|---|---|---|---|---|
| grasp/box | 50 | 12% | 4.9 | @0.06=0.12  @0.03=0.10  @0.02=0.06 |
| grasp/cup | 50 | 40% | 6.6 | @0.06=0.40  @0.03=0.02  @0.02=0.02 |
| reach_with_object/cup | 50 | 98% | 5.2 | @0.1=0.98  @0.06=0.88  @0.03=0.06  @0.015=0.04 |
