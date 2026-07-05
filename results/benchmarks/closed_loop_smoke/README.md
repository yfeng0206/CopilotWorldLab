# Closed-loop smoke benchmark (V-JEPA 2-AC + CEM)

Config: V-JEPA 2-AC ViT-g, samples=200, cem_steps=10, rollout/horizon T=2, topk=10,
maxnorm=0.05 m/axis, bf16, RTX 3090, seed 0. Model sees only image + EE state + goal image
(PLANNING_CAMERA); success judged from hidden MuJoCo truth. V-JEPA plans the coarse motion;
scripted primitives handle only the gripper. Full methodology + logging:
[docs/experiments/closed_loop_benchmark.md](../../../docs/experiments/closed_loop_benchmark.md).

## Success rates (5 trials/task, randomized cube/target)

| task | success | mean V-JEPA final error | read |
|---|---|---|---|
| Reach | **5/5** | 3-7 cm | pure V-JEPA closed-loop, reliable |
| Grasp-Lift | **3/5** | 5.5-7.7 cm | V-JEPA reaches the grasp pose; close+lift grasps ~60% (misses `missed`) |
| Place | **0/5** | **15-18 cm** | vanilla V-JEPA can't precisely place the held cube -- plateaus ~15 cm vs the 6 cm zone (`outside_zone`) |

All 15 trials: `trials_5trial.csv`. Representative rollouts (red=object, blue=EE, green=zone,
+ stats panel): `reach_success.gif`, `grasp_success.gif`, `grasp_missed.gif`,
`place_outside_zone.gif` (+ `*_contact.png`).

Honest boundary: reach is easy, grasp is decent, **place exposes the precision gap** -- the
vanilla baseline the W*/fine-tuning/cross-view improvements (Phases 2-4) must beat. Smoke scale
(5 trials); the full 50-trial precision-curve run is prepared (`--trials 50`).

Reproduce: `python scripts/run_closed_loop_benchmark.py --tasks reach grasp_lift place --trials 5 --tag smoke`
