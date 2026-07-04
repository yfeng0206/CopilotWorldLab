# archive/

Deprecated / historical files kept for provenance but no longer part of the active project.
Nothing here is imported by `src/`, `scripts/`, or the active test suite.

- `src/mujoco_scene.py` — the original `MujocoPilotEnv`: a **kinematic mocap** end-effector env
  (no arm, no IK, no dynamics) over a vial/holder tabletop scene. Superseded by the real
  `FrankaDroidEnv` (`src/envs/franka_droid_env.py`), which drives a physical 7-DoF arm with
  differential IK, contacts and a Robotiq gripper. Kept in case the labware vial/holder scene is
  revived for the downstream application.
- `assets/scene.xml` — the mocap-era vial/holder tabletop MJCF used by `MujocoPilotEnv`. The
  eventual labware-insertion asset; not used by the current Franka benchmark.
- `tests/test_mujoco_env.py`, `tests/test_render.py` — tests for the archived mocap env. Render
  coverage is now provided by `tests/test_franka.py` and `tests/test_bench_env.py`.
- `docs/setup_stage.md` — the pre-experiment setup record (environment, checkpoint load, timing
  baseline). Superseded by [`../docs/architecture.md`](../docs/architecture.md) and
  [`../docs/research_log.md`](../docs/research_log.md).
- `configs/mujoco_pilot.yaml` — the mocap-era scene/render/planner config, not loaded by any live
  code (the generic loader `src/utils/config.py` stays in `src`). Orphaned when the mocap env was
  archived.
- `scripts/franka_smoke_test.py`, `scripts/franka_viewer_demo.py` — one-off dev/smoke scripts for
  the Franka scene; superseded by the test suite (`tests/test_franka.py`, `tests/test_bench_env.py`).
