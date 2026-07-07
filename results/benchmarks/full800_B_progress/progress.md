# Benchmark progress -- full800_B (run 20260706_184713)

_Snapshot: 2026-07-06T21:36:12. Live run, still in progress._

- config: samples **800**, cem_steps 10, horizon T=1, maxnorm **0.075**, camera **B_closer**, dtype bf16
- git commit: `a3b288e79f`
- trials completed: **50**

| task / object | n | success@loosest | mean err (cm) | per-threshold success@x |
|---|---|---|---|---|
| grasp/cup | 50 | 40% | 6.6 | @0.06=0.40  @0.03=0.02  @0.02=0.02 |
