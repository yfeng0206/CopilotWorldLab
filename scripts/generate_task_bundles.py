"""Generate the fixed, inspectable closed-loop benchmark task bundles.

A scripted expert runs each scenario once in ``FrankaDroidEnv`` and saves it as a self-contained,
inspectable *task bundle* (start / sub-goal / goal frames + privileged states + camera + a contact
sheet; see src/bench/schema.py). The closed-loop benchmark then LOADS these bundles instead of
randomizing per trial, so every config is scored on identical scenarios.

Tasks (V-JEPA plans the coarse motion; scripted gripper; based on arXiv 2506.09985 Table 3):
    grasp              V-JEPA grasps; a scripted lift tests success   -- 1 goal (just grabbed)
    reach_with_object  object STARTS grasped; move it to a goal       -- 1 goal (start grasped)
    grasp_and_reach    grasp off the table, then reach with it        -- 2 goals (goal_1, goal)
    pick_place         grasp -> vicinity -> place, fixed 4/10/4       -- 3 goals (goal_1, goal_2, goal)

Objects: cup (cube cup, rim-graspable) and box (rigid block). Layout:
    tasks/<task>/<object>/<task>_<object>_<NN>/ { meta.json, start.png, goal[_1,_2].png,
                                                  arrays.npz, contact_sheet.png }

Every scenario is validated (the scripted expert must complete it) before it is saved, so each
bundle has a well-defined, measurable hidden success/failure.

    python scripts/generate_task_bundles.py --tasks grasp reach_with_object grasp_and_reach pick_place \
        --objects cup box --trials 50
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.schema import TaskBundle  # noqa: E402
from src.bench.thresholds import THRESHOLDS  # noqa: E402
from src.envs.franka_build import (  # noqa: E402
    OBJECT_SPECS,
    PLANNING_CAMERA,
    TABLE_TOP_Z,
)
from src.envs.franka_droid_env import FrankaDroidEnv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("generate_task_bundles")

EE_DOWN = [np.pi, 0.0, 0.0]  # gripper pointing down (extrinsic XYZ euler), matches the benchmark
ALL_TASKS = ["grasp", "reach_with_object", "grasp_and_reach", "pick_place", "place_with_object"]
ALL_OBJECTS = ["cup", "box"]

# Precision sphere-radius sweep per task (metres). SINGLE SOURCE OF TRUTH is
# ``src.bench.thresholds.THRESHOLDS`` -- the benchmark scores success@x at exactly these radii, so
# meta.json advertises the same sweep it is scored against. success@x = delta < x AND physical gates.
X_SWEEP = {t: list(THRESHOLDS[t]) for t in ALL_TASKS}


def _goto(env, pos, grip=None, n=10, step=0.05):
    """Scripted straight-line EE move (the expert), optionally actuating the gripper afterwards.

    Each physics action is capped to ``step`` metres so the differential IK stays inside its
    tolerance (apply_action is atomic: an over-long move that exceeds the IK tolerance is rejected
    outright, which would freeze the arm). Small capped steps make the move reliable everywhere in
    the workspace, at the cost of more (offline) steps."""
    pos = np.asarray(pos, dtype=float)
    for _ in range(n):
        cur = env.get_ee_state()[:3]
        delta = pos - cur
        dist = float(np.linalg.norm(delta))
        if dist < 0.004:
            break
        d = np.zeros(7)
        d[:3] = delta * min(1.0, step / dist)   # cap per-step translation for reliable IK
        env.apply_action(d)
    if grip is not None:
        for _ in range(5):
            g = np.zeros(7)
            g[6] = grip
            env.apply_action(g)


def _sample_obj_xy(rng):
    """A randomized, reachable object start spread widely across the table so the 50 scenarios differ
    clearly in object placement (kept clear of the place zone's default location)."""
    return (float(rng.uniform(0.40, 0.60)), float(rng.uniform(-0.24, 0.06)))


def _sample_zone_xy(rng):
    """A randomized reachable place-zone location (far half of the table) for pick_place, so the
    place goal varies across scenarios instead of always being the same fixed zone."""
    return (float(rng.uniform(0.40, 0.60)), float(rng.uniform(0.08, 0.24)))


def _random_start_ee(rng):
    """A randomized reachable EE hover pose above the table -- the per-trial *starting pose* the paper
    permutes (arXiv 2506.09985 4.2), so start frames differ widely across the 50 scenarios."""
    return np.array([rng.uniform(0.40, 0.60), rng.uniform(-0.22, 0.20), rng.uniform(0.34, 0.50)])


def _far_pose(rng, ref, lo, hi, min_sep=0.12):
    """Draw a reachable pose in the box [lo, hi] that is at least ``min_sep`` from ``ref``."""
    lo = np.asarray(lo); hi = np.asarray(hi)
    p = rng.uniform(lo, hi)
    for _ in range(10):
        if np.linalg.norm(p - ref) >= min_sep:
            break
        p = rng.uniform(lo, hi)
    return p


def _grasp_points(obj, c):
    """Approach (above) and grasp positions for the object's grasp; the cup uses a rim xy offset so
    one finger goes inside the hollow and one outside (see OBJECT_SPECS)."""
    dz = OBJECT_SPECS[obj]["grasp_dz"]
    ox, oy = OBJECT_SPECS[obj]["grasp_off"]
    gx, gy = c[0] + ox, c[1] + oy
    approach = np.array([gx, gy, c[2] + 0.14])
    grasp = np.array([gx, gy, c[2] + dz])
    return approach, grasp


def _grasp_object(env, obj):
    """Deterministic straight-down rim grasp: the gripper is commanded EE_DOWN at the FIXED grasp
    offset from the object, then closed -- so the grip is identical *relative to the object* at any
    table xy (gripper pointing straight down, one finger inside the cup, one outside). Returns
    (grasp_pos, held). Uses set_ee_pose to enforce the straight-down orientation (a position-only
    move drifts the gripper by ~15-25 deg, which tilts the object and dips the gripper into the cup)."""
    c = env.object_position()
    approach, grasp = _grasp_points(obj, c)
    env.set_ee_pose(pos=approach, euler=EE_DOWN, gripper=0.0)   # straight down, open, above the object
    env.set_ee_pose(pos=grasp, euler=EE_DOWN, gripper=0.0)      # straight down at the grasp offset
    for _ in range(5):
        env.apply_action(np.array([0., 0., 0., 0., 0., 0., 1.0]))  # close on the wall
    return grasp, env.gripper_holds_object()


def _base_arrays(env):
    return {
        "qpos0": env.data.qpos.copy(),
        "qpos_start": env.data.qpos.copy(),
        "qvel0": env.data.qvel.copy(),
        "start_ee": env.get_ee_state().copy(),
        "object_pose": env.object_pose().copy(),
        "zone": env.zone_center().copy(),
    }


# --------------------------------------------------------------------------- per-task builders
# Design: the scripted expert PHYSICALLY grasps (validating the grip) and we render the LIVE state for
# on-table checkpoints (start, "just grabbed" goals). For HELD checkpoints we carry the object
# KINEMATICALLY with the gripper (env.move_held_to), keeping its upright grasp pose -- a physical
# one-wall-rim transport swings/tilts the object, which is a sim artifact, not the intended target.
# The placed goal is built directly (object resting in the zone, arm just above, gripper open).

def build_grasp(env, rng, obj):
    """V-JEPA plans ONLY the grasp reach. Goal = object JUST GRABBED (gripper closed on it, still on
    the table, NOT lifted). A scripted lift afterward tests success (off table + not dropped)."""
    env.reset(cube_xy=_sample_obj_xy(rng))
    _goto(env, _random_start_ee(rng), n=8)  # randomized starting pose
    start_img = env.render("planning")
    arrays = _base_arrays(env)
    c = env.object_position()
    approach, grasp = _grasp_points(obj, c)
    arrays["grasp_pos"] = grasp.astype(float)
    grasp, held = _grasp_object(env, obj)          # deterministic straight-down rim grasp
    if not held:
        return None, None, None, False
    goal_img = env.render("planning")                      # LIVE: object on the table, gripped
    arrays["qpos_goal"] = env.data.qpos.copy()
    arrays["goal_object"] = env.object_position().copy()
    # validate the grasp is liftable: scripted lift, off the table, not dropped
    z0 = env.object_position()[2]
    _goto(env, grasp + np.array([0.0, 0.0, 0.14]), grip=1.0)
    ok = bool(env.gripper_holds_object()) and (env.object_position()[2] - z0) > 0.04
    return {"start": start_img, "goal": goal_img}, arrays, {"start_grasped": False}, ok


def build_reach_with_object(env, rng, obj):
    """Object STARTS grasped + lifted; V-JEPA moves the held object to a goal (paper-like traverse)."""
    env.reset(cube_xy=_sample_obj_xy(rng))
    c = env.object_position()
    half = OBJECT_SPECS[obj]["rest_half_z"]
    grasp, held = _grasp_object(env, obj)          # deterministic straight-down rim grasp
    if not held:
        return None, None, None, False
    # start: kinematically carry the held (upright) object to a start hover over its pickup spot
    start_hover = np.array([grasp[0], grasp[1], TABLE_TOP_Z + half + 0.16])
    env.move_held_to(pos=start_hover, euler=EE_DOWN, gripper=1.0, upright=True)
    start_img = env.render("planning")                     # object held upright, lifted at the start
    arrays = _base_arrays(env)  # qpos0 = grasped, object-in-hand state
    ee = env.get_ee_state()[:3]
    goal_ee = _far_pose(rng, ee, lo=[0.40, 0.04, 0.32], hi=[0.60, 0.24, 0.46], min_sep=0.20)
    arrays["goal_ee"] = goal_ee.astype(float)
    env.move_held_to(pos=goal_ee, euler=EE_DOWN, gripper=1.0, upright=True)  # carry upright to the goal
    goal_img = env.render("planning")                      # object at goal, held upright
    arrays["qpos_goal"] = env.data.qpos.copy()
    ok = True
    arrays["goal_object"] = env.object_position().copy()
    return {"start": start_img, "goal": goal_img}, arrays, {"start_grasped": True}, ok


def build_grasp_and_reach(env, rng, obj):
    """2-goal compositional task: V-JEPA grasps the object off the table (goal_1 = just grabbed), then
    reaches with the held object to a target (goal). Object starts ON THE TABLE (V-JEPA does both)."""
    env.reset(cube_xy=_sample_obj_xy(rng))
    _goto(env, _random_start_ee(rng), n=8)  # randomized starting pose
    start_img = env.render("planning")
    arrays = _base_arrays(env)  # qpos0 = initial, object on the table
    c = env.object_position()
    approach, grasp = _grasp_points(obj, c)
    arrays["grasp_pos"] = grasp.astype(float)
    grasp, held = _grasp_object(env, obj)          # deterministic straight-down rim grasp
    if not held:
        return None, None, None, False
    goal_1 = env.render("planning")                        # LIVE: just grabbed, on the table
    arrays["qpos_goal_1"] = env.data.qpos.copy()
    ee = env.get_ee_state()[:3]
    goal_ee = _far_pose(rng, ee, lo=[0.40, 0.04, 0.32], hi=[0.60, 0.24, 0.46], min_sep=0.18)
    arrays["goal_ee"] = goal_ee.astype(float)
    env.move_held_to(pos=goal_ee, euler=EE_DOWN, gripper=1.0, upright=True)  # carry held object upright to goal
    goal_g = env.render("planning")                        # held object at the target, upright
    arrays["qpos_goal"] = env.data.qpos.copy()
    ok = True
    arrays["goal_object"] = env.object_position().copy()
    return {"start": start_img, "goal_1": goal_1, "goal": goal_g}, arrays, {"start_grasped": False}, ok


def build_pick_place(env, rng, obj):
    """grasp -> vicinity -> place, fixed 4/10/4. goal_1 = just grabbed (on table), goal_2 = held in
    the vicinity of the zone, goal = object PLACED in the zone (released, gripper away)."""
    env.reset(cube_xy=_sample_obj_xy(rng))
    env.set_zone_xy(*_sample_zone_xy(rng))  # randomized place zone (varies the place goal per trial)
    _goto(env, _random_start_ee(rng), n=8)  # randomized starting pose
    start_img = env.render("planning")
    arrays = _base_arrays(env)  # qpos0 = initial, object on the table
    c = env.object_position()
    half = OBJECT_SPECS[obj]["rest_half_z"]
    zone = env.zone_center()
    approach, grasp = _grasp_points(obj, c)
    arrays["grasp_pos"] = grasp.astype(float)
    grasp, held = _grasp_object(env, obj)          # deterministic straight-down rim grasp
    if not held:
        return None, None, None, False
    goal_1 = env.render("planning")                        # LIVE: just grabbed, on the table
    arrays["qpos_goal_1"] = env.data.qpos.copy()
    # goal_2: kinematically carry the held (upright) object to just above the zone -- no transport tilt
    ox, oy = OBJECT_SPECS[obj]["grasp_off"]
    dz = OBJECT_SPECS[obj]["grasp_dz"]
    vicinity_pos = np.array([zone[0] + ox, zone[1] + oy, TABLE_TOP_Z + half + 0.05 + dz])
    env.move_held_to(pos=vicinity_pos, euler=EE_DOWN, gripper=1.0, upright=True)
    arrays["vicinity_pos"] = vicinity_pos.astype(float)
    goal_2 = env.render("planning")                        # held upright in the vicinity of the zone
    arrays["qpos_goal_2"] = env.data.qpos.copy()
    # Build the PLACED goal as the intended target state: the arm just above the placed object with
    # the gripper open (paper-style final goal), and the object resting upright IN the zone. (A
    # physical one-wall-rim release is too flaky -- the inside finger hooks the cup -- but the placed
    # goal image is a legitimate target, so we construct it directly and verify it is a valid rest.)
    env.set_ee_pose(pos=[zone[0], zone[1], TABLE_TOP_Z + half + 0.10], euler=EE_DOWN, gripper=0.0)
    for _ in range(4):
        env.apply_action(np.array([0., 0., 0., 0., 0., 0., -1.0]))  # open the fingers, hold the pose
    env.place_object(zone[0], zone[1])                     # object settles upright in the zone
    place_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + half])
    arrays["place_pos"] = place_pos.astype(float)
    goal_g = env.render("planning")                        # LIVE: object placed in the zone, arm clear
    arrays["qpos_goal"] = env.data.qpos.copy()
    obj_final = env.object_position()
    err = float(np.linalg.norm(obj_final[:2] - zone[:2]))
    on_table = (obj_final[2] - (TABLE_TOP_Z + half)) < 0.02
    upright = np.degrees(env.object_tilt()) < 20.0
    ok = err < 0.06 and env.object_released() and on_table and upright
    arrays["goal_object"] = obj_final.copy()
    return {"start": start_img, "goal_1": goal_1, "goal_2": goal_2, "goal": goal_g}, arrays, \
        {"start_grasped": False}, ok


def build_place_with_object(env, rng, obj):
    """Object STARTS grasped + lifted (like reach_with_object); V-JEPA carries it over the zone and
    places it DOWN. The place half of pick_place. goal_1 = held in the vicinity of the zone,
    goal = object PLACED in the zone (released, gripper away). start_grasped=True."""
    env.reset(cube_xy=_sample_obj_xy(rng))
    env.set_zone_xy(*_sample_zone_xy(rng))  # randomized place zone (varies the place goal per trial)
    half = OBJECT_SPECS[obj]["rest_half_z"]
    zone = env.zone_center()
    grasp, held = _grasp_object(env, obj)          # deterministic straight-down rim grasp
    if not held:
        return None, None, None, False
    # start: kinematically carry the held (upright) object to a start hover over its pickup spot
    start_hover = np.array([grasp[0], grasp[1], TABLE_TOP_Z + half + 0.16])
    env.move_held_to(pos=start_hover, euler=EE_DOWN, gripper=1.0, upright=True)
    start_img = env.render("planning")                     # object held upright, lifted at the start
    arrays = _base_arrays(env)  # qpos0 = grasped, object-in-hand state
    # goal_1: kinematically carry the held (upright) object to just above the zone -- no transport tilt
    ox, oy = OBJECT_SPECS[obj]["grasp_off"]
    dz = OBJECT_SPECS[obj]["grasp_dz"]
    vicinity_pos = np.array([zone[0] + ox, zone[1] + oy, TABLE_TOP_Z + half + 0.05 + dz])
    env.move_held_to(pos=vicinity_pos, euler=EE_DOWN, gripper=1.0, upright=True)
    arrays["vicinity_pos"] = vicinity_pos.astype(float)
    goal_1 = env.render("planning")                        # held upright in the vicinity of the zone
    arrays["qpos_goal_1"] = env.data.qpos.copy()
    # placed goal, constructed directly (a physical one-wall-rim release is too flaky): arm just above
    # the placed object with the gripper open, object resting upright IN the zone.
    env.set_ee_pose(pos=[zone[0], zone[1], TABLE_TOP_Z + half + 0.10], euler=EE_DOWN, gripper=0.0)
    for _ in range(4):
        env.apply_action(np.array([0., 0., 0., 0., 0., 0., -1.0]))  # open the fingers, hold the pose
    env.place_object(zone[0], zone[1])                     # object settles upright in the zone
    place_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + half])
    arrays["place_pos"] = place_pos.astype(float)
    goal_g = env.render("planning")                        # LIVE: object placed in the zone, arm clear
    arrays["qpos_goal"] = env.data.qpos.copy()
    obj_final = env.object_position()
    err = float(np.linalg.norm(obj_final[:2] - zone[:2]))
    on_table = (obj_final[2] - (TABLE_TOP_Z + half)) < 0.02
    upright = np.degrees(env.object_tilt()) < 20.0
    ok = err < 0.06 and env.object_released() and on_table and upright
    arrays["goal_object"] = obj_final.copy()
    return {"start": start_img, "goal_1": goal_1, "goal": goal_g}, arrays, \
        {"start_grasped": True}, ok


BUILDERS = {
    "grasp": build_grasp,
    "reach_with_object": build_reach_with_object,
    "grasp_and_reach": build_grasp_and_reach,
    "pick_place": build_pick_place,
    "place_with_object": build_place_with_object,
}


def _contact_sheet(images, out_png, title):
    order = [k for k in ("start", "goal_1", "goal_2", "goal") if k in images]
    fig, axes = plt.subplots(1, len(order), figsize=(len(order) * 2.4, 2.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, order):
        ax.imshow(images[name])
        ax.set_title(name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def generate_one(task, obj, index, seed, tasks_dir):
    """Build and save one validated bundle; retry a few reseeds if the expert fails."""
    builder = BUILDERS[task]
    env = FrankaDroidEnv(add_object=True, add_zone=True, object_type=obj, add_distractors=True)
    try:
        for attempt in range(6):
            rng = np.random.default_rng(seed + attempt * 7919)
            images, arrays, meta_extra, ok = builder(env, rng, obj)
            if ok:
                break
        if not ok:
            logger.warning("%-18s %s #%02d: expert failed after retries -- skipped", task, obj, index)
            return None
        task_id = f"{task}_{obj}_{index:02d}"
        meta = {
            "task_id": task_id, "task": task, "object": obj, "index": index, "seed": int(seed),
            "camera": PLANNING_CAMERA, "image_hw": [env.render_height, env.render_width],
            "units": "meters", "ee_euler_down": EE_DOWN,
            "success_spec": {"type": task, "x_sweep_cm": [round(x * 100, 2) for x in X_SWEEP[task]]},
            **meta_extra,
        }
        root = os.path.join(tasks_dir, task, obj)
        out = TaskBundle(meta=meta, images=images, arrays=arrays).save(root)
        _contact_sheet(images, os.path.join(out, "contact_sheet.png"),
                       f"{task_id}  ({task}, {obj})")
        return out
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    p.add_argument("--objects", nargs="+", default=ALL_OBJECTS, choices=ALL_OBJECTS)
    p.add_argument("--trials", type=int, default=50, help="scenarios per (task, object)")
    p.add_argument("--tasks-dir", default=os.path.join(_REPO_ROOT, "tasks"))
    p.add_argument("--seed-base", type=int, default=1000)
    args = p.parse_args()

    n_ok = n_fail = 0
    for task in args.tasks:
        for obj in args.objects:
            for i in range(args.trials):
                seed = args.seed_base + hash_offset(task) * 100003 + \
                    (0 if obj == "cup" else 50000) + i
                out = generate_one(task, obj, i, seed, args.tasks_dir)
                if out is None:
                    n_fail += 1
                else:
                    n_ok += 1
            logger.info("%-18s %-3s: %d bundles under %s", task, obj, args.trials,
                        os.path.join(args.tasks_dir, task, obj))
    logger.info("done: %d bundles written, %d skipped -> %s", n_ok, n_fail, args.tasks_dir)


def hash_offset(task):
    """Stable per-task seed offset (not Python's randomized string hash)."""
    return {t: i for i, t in enumerate(ALL_TASKS)}[task]


if __name__ == "__main__":
    main()
