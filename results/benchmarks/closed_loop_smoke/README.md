# Closed-loop smoke benchmark (V-JEPA 2-AC + CEM)

Config: samples=200, cem_steps=10, rollout/horizon T=2, topk=10, maxnorm=0.05 m/axis, bf16, RTX 3090.
Model sees only image + EE state + goal image (PLANNING_CAMERA); success from hidden MuJoCo truth.
Scripted primitives handle the gripper (align/close/lift/open); V-JEPA plans the coarse motion.

| task | trial | success | key metric |
|---|---|---|---|
| reach | 0 | PASS | EE-to-goal distance 0.021 m (< 0.05) |
| grasp_lift | 0 | PASS | object dz +0.075 m, held, tilt 17 deg, settled |
| place | 0 | PASS | object 0.004 m from zone center, upright, released |

Artifacts (per task): `<task>_t0.gif` (rollout) and `<task>_t0_contact.png` (annotated frames:
red=object, blue=EE, green=zone circle, + stats panel). Smoke scale (1 trial each) -- not a
success-rate result. Reproduce: `python scripts/run_closed_loop_benchmark.py --tasks reach grasp_lift place --trials 1`
