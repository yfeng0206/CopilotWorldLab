# Closed-Loop Task-Success Benchmark — Implementation Plan

## Why this exists

The V-JEPA 2 paper's headline **robot** metric is **closed-loop task-success rate** on a real
Franka (Reach 100%, Grasp Cup 65% / Box 25%, Reach-with-object ~75%, Pick-and-place Cup 80% /
Box 65%), plus planning efficiency (CEM samples/time vs Cosmos) and an energy-landscape sanity
check. Our current DROID transition-scoring result (rank_frac 0.820) is a **world-model /
transition sanity** number only — it does **not** measure task success. This plan builds the
success-rate benchmark so "improvement" is measured the way the paper measures it.

**Core rule (matches mainstream robot benchmarks).** The model sees only observations:
RGB image, EE state, and a goal image. The evaluator uses **privileged** MuJoCo truth: object
pose, target pose, contacts, velocities, tilt — to compute a **hidden** success label the model
never sees.

**Data policy (user directive 2026-07-04):** use established robomimic data if at all possible;
only build our own MuJoCo mini-suite if robomimic is truly unusable. Rendering robomimic **raw**
demos on Windows is verified working (plain MuJoCo + patched robosuite assets; no robosuite
runtime), so robomimic is the primary source.

---

## 1. Staged implementation plan

| Stage | Goal | Primary scripts |
|---|---|---|
| **0. Data / scene** | Reliable start/goal images, states, actions, hidden success labels; **visual approval before any benchmarking** | `scripts/build_own_tasks.py` (scripted expert in our env) + `src/bench/schema.py`. Robomimic render (`render_robomimic_task.py`) is a **reference/image source only**, not the closed-loop benchmark. |
| **1. Success checks** | Exact hidden success functions (Reach/Touch/Grasp-Lift/Place) on privileged state | `src/bench/success.py` + `tests/test_success.py` |
| **2. Vanilla benchmark** | Run CEM closed loop, no fine-tuning; N trials/task/difficulty; log everything | `scripts/run_closed_loop_benchmark.py` on our own `FrankaDroidEnv` (add_object/add_zone); reuses the `scripts/cem_reach_loop.py` CEM |
| **3. Diagnose failures** | Classify every failure (frame error, missed grasp, pushed, slipped, tipped, outside zone, energy/physics mismatch) | `scripts/diagnose_failures.py` |
| **4. Improvements** | W* frame calibration, predictor fine-tune (frozen encoder), energy-gate calibration | `scripts/fit_wstar.py`, `scripts/finetune_predictor.py`, `scripts/energy_gate_roc.py` |
| **5. Final comparison** | Report Vanilla vs +W* vs +Fine-tune table + figures | `scripts/make_benchmark_report.py` |

**Sequencing:** Stage 0 → **user visual approval** → Stage 1 (with unit tests) → Stage 2 vanilla
baseline → Stage 3 → Stage 4 (only after a solid vanilla baseline) → Stage 5. Each coding step is
committed and paused for manual audit before the next.

**The closed-loop crux (Stage 2) — decision: our own honest env, reproduce the paper's MPC.**
We are **not** using robosuite for the closed-loop benchmark. robosuite 1.5.2 calls the old
`mujoco.mj_fullM(model, dst, qM)` (2-array) signature at
`controllers/parts/controller.py:227`, but mujoco >= ~3.a changed the Python binding to
`mj_fullM(m, d, dst)`; with our mujoco 3.10 `env.step` raises `TypeError` (re-confirmed
2026-07-04). It would run only in a **separate venv with a pinned older mujoco** (~3.3.0 still has
the 2-array binding), which we are choosing **not** to maintain. Instead (user directive: honest,
established-style, not a toy, no hacking) we build our **own** small MuJoCo grasp/place env with
**real physics and hidden privileged success**, and reproduce the paper's planning loop faithfully.

### Verified planning config (primary source: Meta's released code)

From `third_party/vjepa2/notebooks/utils/world_model_wrapper.py` (the `WorldModel` MPC defaults)
and `.../mpc_utils.py::cem` — the exact procedure we reproduce:

| param | released code (verified) | paper text | our runs |
|---|---|---|---|
| horizon `rollout` (T) | **2** | horizon **1** reported (to reconcile) | ablate T=1 vs T=2 |
| population `samples` | **400** | ~800 | staged 100 -> 200 -> 800 |
| CEM iterations `cem_steps` | **10** | 10 | 10 |
| `topk` | 10 | 10 | 10 |
| `maxnorm` (per-axis action clip) | 0.05 m | 0.05 m | 0.05 m |
| momentum (mean/std) | 0.15 | - | 0.15 |
| objective | mean-L1 in **layer-norm'd** latent space | L1 energy | same |
| replan | receding horizon (re-encode, replan) | receding horizon | same |

**Horizon caveat (honest):** the only value we have directly verified is the released inference
wrapper default, `rollout=2`. A prior read of the paper suggested a planning **horizon of 1**; we
did not re-verify the paper text this session (the ar5iv/arxiv HTML would not paginate to the
planning section). So we do **not** claim "the paper uses T=2." We treat T as a config we set and
**ablate T=1 vs T=2**, and will reconcile against the paper before any headline claim.

CEM samples xyz + gripper (rotation zeroed); the world-model rollout predicts `rollout` steps and
compares the **final** predicted latent to the goal latent; top-k update; the returned action
trajectory's first step(s) are executed, then re-plan. This is exactly what
`scripts/cem_reach_loop.py` already calls, so the benchmark reuses it.

### Staged run cadence (user directive)

1. **Validate the loop** — 3 tasks x 5-8 actions, small `samples`; confirm the closed loop runs
   and latent energy / distance decrease. (Not a result, a wiring check.)
2. **Sweep** — 20 trials at `samples=100` and 20 at `samples=200` (randomized init); compare
   success / energy / CEM time. Pick the operating point the 3090 can afford.
3. **Final** — the selected run at `samples=800` (T=1 and/or T=2) for the headline vanilla number.

Improvement (the goal) comes from W* frame calibration and predictor fine-tuning re-run on the
**same** tasks/seeds/config (Stage 4).

**Env for Stage 2:** extend our MuJoCo Franka scene with graspable objects + a target zone
(honest physics, not a toy), render the validated planning camera, and get image goals from a
**scripted expert** execution (move-to-grasp / close / lift / move-to-place / open) whose key
frames are the sub-goal images. Success is judged only by hidden privileged state (Section 4),
never by matching the scripted trajectory.

---

## 2. File / script names

```
src/bench/schema.py            TaskBundle dataclass + save/load (meta.json + arrays.npz + PNGs)
src/bench/success.py           success functions: reach, touch, grasp_lift, place (privileged state)
src/envs/robomimic_render.py   patch_asset_paths, RobomimicDemoRenderer (raw-state -> image + poses)
src/envs/robomimic_scene.py    steppable plain-MuJoCo wrapper of a robomimic model_file (Stage 2)
scripts/render_robomimic_task.py   Stage 0: render demos -> task bundles + contact sheet + GIF
scripts/build_mujoco_tasks.py      Stage 0 fallback: custom MuJoCo mini-suite tasks
scripts/run_closed_loop_benchmark.py  Stage 2: CEM closed loop over N trials, hidden success, logs
scripts/diagnose_failures.py       Stage 3: failure classification from trial logs
scripts/fit_wstar.py               Stage 4: fit/freeze the App. B.4 view-relative action frame
scripts/finetune_predictor.py      Stage 4: predictor fine-tune (frozen encoder)
scripts/energy_gate_roc.py         Stage 4: does latent energy predict failure? ROC-AUC / FAR
scripts/make_benchmark_report.py   Stage 5: final comparison table + figures
tests/test_success.py              unit tests for success functions
tests/test_robomimic_render.py     patch_asset_paths + bundle round-trip (skipped if data absent)
```

Artifacts: `tasks/<task_id>/...` bundles (gitignored, large), `results/benchmarks/closed_loop/`
(committed summary CSV/JSON + figures + a few small contact sheets for provenance).

---

## 3. Data schema (saved benchmark task)

A task **bundle** is a directory `tasks/<task_id>/`:

```
tasks/<task_id>/
  meta.json          # everything the evaluator + planner need (below)
  start.png          # observation at t0 (planner input)
  goal.png           # goal image (planner target); goal_1.png, goal_2.png for multi-stage
  arrays.npz         # start_state, goal_state, object_state, target_state, ...
  model.xml          # patched MJCF for closed-loop stepping (Option A)
  contact_sheet.png  # visual check: sampled trajectory with state/action overlay
  rollout.gif        # visual check: full demo playback
```

`meta.json`:
```json
{
  "task_id": "lift_ph_demo0_d0",
  "task_type": "grasp_lift",              // reach | touch | grasp_lift | place
  "difficulty": "easy",                    // or a radius/threshold label
  "source": "robomimic/lift/ph/demo_0",
  "camera": "agentview",
  "image_hw": [256, 256],
  "fps": 20,
  "units": "meters",
  "robot_ee_convention": "xyz+euler+gripper (7-D)",
  "object_body": "cube_main",
  "target": {"type": "zone", "center": [x,y,z], "radius": 0.06}  // place; null for reach/lift
  "success_spec": {                        // thresholds the evaluator uses (Section 4)
    "type": "grasp_lift",
    "lift_dz": 0.04, "grasp_radius": 0.05, "tilt_max_deg": 30, "v_settle": 0.05
  },
  "seed": 0
}
```

`arrays.npz` (all float32 unless noted):
```
start_state   [7]     EE (x,y,z,roll,pitch,yaw,gripper) at t0
goal_state    [7]     EE at goal
object_state  [7]     object pose (x,y,z, quat wxyz) at t0
goal_object_state [7] object pose at goal (for lift/place references)
target_state  [7|4]   target pose or (center xyz + radius) for place; nan for reach/lift
qpos0         [nq]    full MuJoCo qpos at t0 (to reset the steppable env)
qvel0         [nv]    full MuJoCo qvel at t0
actions       [T,7]   demo actions (reference / for scripted lift-place phases)
```

`TaskBundle.save(dir)` / `TaskBundle.load(dir)` in `src/bench/schema.py`; the planner reads only
`start.png`, `goal.png`, `start_state`; the evaluator reads privileged `object_state`,
`target_state`, contacts/velocities from the live sim.

---

## 4. Exact success functions (privileged MuJoCo state)

All operate on the **evaluator's** live sim after the controlled/scripted phase settles. Notation:
`EE` = end-effector site pos; `obj_p`, `obj_R` = object body position / rotation; `obj_up` =
`obj_R @ [0,0,1]`; `tilt = acos(clamp(obj_up . [0,0,1]))`; `v_obj` = object linear speed.

**Reach** (`reach`):
```
success = ||EE - target_p|| < tau_reach
tau_reach in {0.05, 0.03, 0.015} m   (easy -> hard)
```

**Touch** (`touch`):
```
contact = any MuJoCo contact pair between a gripper geom and the target object geom
success = contact AND ||obj_p - obj_p_0|| < move_tol      (touched, not knocked away)
move_tol = 0.03 m ; alt. proximity form: ||EE - obj_p|| < 0.02 m
```

**Grasp-Lift** (`grasp_lift`) — after the model reaches the grasp goal, scripted: close gripper,
lift +Δz (0.05–0.10 m) over K steps, settle:
```
lifted   = (obj_z_final - obj_z_0) > lift_dz            # default 0.04 m
held     = ||obj_xy_final - EE_xy_final|| < grasp_radius # default 0.05 m (stayed with gripper)
upright  = tilt < tilt_max                               # default 30 deg
stable   = v_obj < v_settle                              # default 0.05 m/s after settle
success  = lifted AND held AND upright AND stable
```
Failure typing: `dropped` (not lifted & obj_z low), `slipped` (lifted earlier but not held),
`tipped` (tilt >= tilt_max), `missed` (obj never moved), `pushed` (obj moved but not lifted).

**Place** (`place`) — object held; model plans to the place goal image; open gripper; settle:
```
in_zone  = ||obj_xy_final - target_xy|| < zone_radius    # {0.10, 0.06, 0.03, 0.015}
upright  = tilt < tilt_max                               # default 25 deg
stable   = v_obj < v_settle                              # default 0.05 m/s
released = gripper open AND no persistent gripper<->object contact (object resting on surface)
success  = in_zone AND upright AND stable AND released
```
Failure typing: `outside_zone`, `tipped`, `unstable`, `still_attached`.

Difficulty is swept by tightening the threshold (`tau_reach`, `zone_radius`) or object
(cube -> cylinder -> box). Every trial also records the continuous quantities (distance, tilt,
velocity, obj_dz) so ROC/gate analysis (Stage 4) is possible.

---

## 5. robomimic as a REFERENCE / image source (NOT the closed-loop benchmark)

**Clarification (do not conflate):** the closed-loop task-success benchmark runs on **our own
honest MuJoCo env** (Section on the crux above), because robosuite's runtime is blocked and we
want full control of physics + hidden success. robomimic here is only: (a) an **established
real-demo image/pose reference** to sanity-check our scene/camera against real Lift/Can/Square,
and (b) available via Windows-safe raw-state rendering. It does **not** provide the closed-loop
success numbers. The DROID transition benchmark (transition_scoring.md) is the established
real-data sanity; the cube tasks are the controlled-physics closed-loop benchmark.

**Robomimic reference order (rendering only):** Lift (grasp-lift, cube) → Can (pick-and-place) →
Square (peg/placement). Start with **Lift**.

**Datasets (already downloaded):** `data/robomimic/v1.5/<task>/ph/demo_v15.hdf5`. Each
`data/demo_i` has `states` [T,32] (flattened `[time, qpos, qvel]`), `actions` [T,7] (OSC deltas),
and attr `model_file` (robosuite MJCF XML). No images are stored — we render them.

**Safe render (verified, no robosuite runtime, no dynamics stepping):**
1. Read `model_file` XML from the demo attrs.
2. **Patch asset paths**: the XML embeds the collector's absolute paths
   (`/home/.../robosuite/models/assets/...`). Regex-rewrite any path ending in
   `robosuite/models/assets/` to the local install
   (`os.path.dirname(robosuite.__file__)/models/assets`). (robosuite's `edit_model_xml` does the
   same; we replicate it without instantiating an env.)
3. `m = mujoco.MjModel.from_xml_string(patched)`, `data = mujoco.MjData(m)`.
4. Per frame `t`: `data.time, data.qpos, data.qvel = split(states[t])`; `mujoco.mj_forward`;
   `mujoco.Renderer(m,H,W).update_scene(data, camera=cam).render()`. **Only `mj_forward`, never
   `mj_step`** for rendering — dynamics stepping is what triggers `mj_fullM`.
5. Privileged truth for labels comes from the same `data`: object body pose (`data.xpos/xquat`),
   EE site pos, `data.contact`, `data.cvel`/`qvel`.

**Cameras available:** `frontview, birdview, agentview, sideview, robot0_robotview,
robot0_eye_in_hand`. For planning we want a consistent exocentric view; default `agentview`
(robomimic's canonical policy camera). The contact sheet renders 2 candidate cameras so the user
can pick the planning camera (our ablation says an over-the-shoulder az45_el45-like view is best,
so a custom free camera is also an option).

**Fallback (only if robomimic proves unusable):** `scripts/build_mujoco_tasks.py` builds a custom
MuJoCo mini-suite (Reach target, Touch box, Grasp-lift cube/cylinder, Place into zone) on
`FrankaDroidEnv`. Same schema, same success functions.

---

## 6. What counts as a valid vanilla V-JEPA baseline

A vanilla baseline is valid only if **all** hold:
1. **No fine-tuning, no W*** — stock checkpoint, stock predictor.
2. **Task passed human visual check** (Stage 0 contact sheet/GIF approved) before running.
3. **Success = hidden sim truth** (Section 4), never latent energy.
4. **Reach sanity ≈ 100%** on the easy threshold (matches the paper; if Reach fails, the
   camera/action interface is wrong, not the task).
5. **Frozen, logged protocol**: fixed CEM budget (samples, cem_steps, rollout horizon), fixed
   planning camera, fixed thresholds, recorded seeds; N ≥ 20 (prefer 50) randomized trials with
   randomized object/start/target positions.
6. **Randomization is real** — start/object/target sampled per trial, not the demo's fixed pose.
7. Report `success_rate` with a **Wilson 95% CI** (N is small).

The vanilla numbers become the fixed reference; +W* and +Fine-tune are re-run on the **exact**
same tasks, seeds, budget, and thresholds.

---

## 7. Metrics for the final comparison

Per trial log: `success` (bool), `final_distance` (m), `steps_to_success`, `final_latent_energy`,
`cem_time_s`, `cem_samples`, `failure_type`, plus continuous `obj_dz`, `tilt`, `v_obj`.

Per task/difficulty aggregate:
- **success_rate** (primary) + Wilson 95% CI
- mean **final_distance**, mean **steps_to_success** (successes only)
- mean **CEM time / samples** (planning efficiency, vs the paper's Cosmos comparison)
- **failure_type** distribution
- **energy gate**: ROC-AUC of `final_latent_energy` predicting success, and the **false-accept
  rate** at the chosen operating point (the project's confidence-gate metric)

Final table:

| Task | Difficulty | Vanilla | +W* | +Fine-tune | Notes |
|---|---|---|---|---|---|
| Reach | 5 cm | | | | sanity ~100% |
| Reach | 3 cm | | | | |
| Reach | 1.5 cm | | | | |
| Grasp-lift | easy (cube) | | | | |
| Place | 10 cm | | | | |
| Place | 6 cm | | | | |
| Place | 3 cm | | | | |

Improvement is a positive delta in success_rate (and/or fewer CEM samples for equal success, or
better energy-gate ROC-AUC) on identical tasks — a falsifiable, established-data result.

---

## Status

Direction (2026-07-04): reproduce the paper's closed-loop **task-success** metric honestly on our
**own** small MuJoCo env (real physics + hidden privileged success), **not** robosuite and **not**
a toy. Task set: **Reach + Grasp-lift (cube) + Pick-and-place (cube into zone)**.

Done:
- Paper MPC config **verified** from Meta's released code (T=2, samples=400/paper~800,
  cem_steps=10, topk=10, maxnorm=0.05, momentum 0.15; see the table above).
- robosuite closed-loop **dropped** (confirmed `mj_fullM` 2-array call at `controller.py:227`;
  lessons #11).
- Own-env foundation **built + validated**: `src/envs/franka_build.py` adds a 4 cm graspable
  free-joint cube (`add_object`) + place-zone marker (`add_zone`); `FrankaDroidEnv` gains
  `add_object/add_zone`, cube placement/settle on `reset(cube_xy)`, and privileged truth
  (`object_pose/position/speed/tilt`, `zone_center`, `gripper_holds_object`). `object_speed`
  uses `mj_objectVelocity` (validated == cvel[3:] on a known shove). Physics checked:
  cube rests stably; a scripted descend->close->lift grasps and lifts it (dz +0.14 m, held;
  contacts are `2f85_{left,right}_pad` <-> cube; tilt ~2 deg, speed ~0.005 m/s).
- Hidden **success functions implemented** (`src/bench/success.py`: reach/touch/grasp_lift/place,
  pure + unit-tested) with thresholds calibrated from scripted rollouts.
- Tests: **40 pass** (added `tests/test_success.py` unit tests + `tests/test_bench_env.py`
  scripted grasp-lift physics regression). Defaults keep the arm-only env unchanged.

**Two distinct benchmarks (do not conflate):** (1) **DROID transition scoring** = *established
real-data* world-model sanity (rank_frac; transition_scoring.md); (2) **our MuJoCo cube tasks** =
*controlled closed-loop physics* task-success benchmark (this doc). The cube tasks are our own
honest env (real physics + hidden success), not an established suite -- reported as such.

Notes for the runner (from audit): `reset(cube_xy=None)` is deterministic by default; the
benchmark **randomizes cube (and place-target) positions per trial** via `cube_xy`. Success
thresholds (`SUCCESS_DEFAULTS`) are calibrated against scripted-expert distributions before any
headline success-rate is reported.

Robomimic raw-render (Stage 0, `render_robomimic_task.py`) stays as an established-data reference
and image source, but the closed-loop success runs in our own env.

Next:
1. Scripted expert (grasp-lift **and** place) -> capture sub-goal images (reach/grasp/lift/place)
   -> task bundles + contact sheet + GIF for **user visual approval**.
2. `scripts/run_closed_loop_benchmark.py` wiring V-JEPA CEM at the paper config, using the
   implemented `src/bench/success.py` for hidden success -> staged runs (validate 3x5-8 ->
   20 trials @100/@200 -> final @800). Reconcile the T=1/T=2 horizon against the paper.

References: [benchmark_plan.md](benchmark_plan.md), [transition_scoring.md](transition_scoring.md),
[cem_closed_loop.md](cem_closed_loop.md), [../lessons_learned.md](../lessons_learned.md) #11/#19.
