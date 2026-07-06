"""Generate the fixed, inspectable closed-loop benchmark task bundles.

A scripted expert runs each scenario once in ``FrankaDroidEnv`` and saves it as a self-contained,
inspectable *task bundle* (start / sub-goal / goal frames + privileged states + camera + a contact
sheet; see src/bench/schema.py). The closed-loop benchmark then LOADS these bundles instead of
randomizing per trial, so every config is scored on identical scenarios.

Tasks (paper's four robot tasks, arXiv 2506.09985 Table 3):
    reach              EE to a goal pose (object in scene)           -- 1 goal image
    grasp              reach the grasp pose; scripted close + lift   -- 1 goal image
    reach_with_object  object STARTS grasped; move it to a goal      -- 1 goal image (start grasped)
    pick_place         grasp -> vicinity -> place, fixed 4/10/4      -- goal_1, goal_2, goal

Objects: cup (rim-graspable) and box (rigid block). Layout:
    tasks/<task>/<object>/<task>_<object>_<NN>/ { meta.json, start.png, goal[_1,_2].png,
                                                  arrays.npz, contact_sheet.png }

Every scenario is validated (the scripted expert must complete it) before it is saved, so each
bundle has a well-defined, measurable hidden success/failure.

    python scripts/generate_task_bundles.py --tasks reach grasp reach_with_object pick_place \
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
from src.envs.franka_build import (  # noqa: E402
    OBJECT_SPECS,
    PLANNING_CAMERA,
    TABLE_TOP_Z,
)
from src.envs.franka_droid_env import FrankaDroidEnv  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("generate_task_bundles")

EE_DOWN = [np.pi, 0.0, 0.0]  # gripper pointing down (extrinsic XYZ euler), matches the benchmark
ALL_TASKS = ["reach", "grasp", "reach_with_object", "pick_place"]
ALL_OBJECTS = ["cup", "box"]

# Precision sphere-radius sweep per task (metres); success@x = delta < x AND physical gates.
X_SWEEP = {
    "reach": [0.05, 0.03, 0.015],
    "grasp": [0.06, 0.03, 0.02],
    "reach_with_object": [0.06, 0.03, 0.015],
    "pick_place": [0.10, 0.06, 0.03, 0.015],
}


def _goto(env, pos, grip=None, n=6):
    """Scripted straight-line EE move (the expert), optionally actuating the gripper afterwards."""
    for _ in range(n):
        cur = env.get_ee_state()[:3]
        d = np.zeros(7)
        d[:3] = np.asarray(pos, dtype=float) - cur
        env.apply_action(d)
    if grip is not None:
        for _ in range(4):
            g = np.zeros(7)
            g[6] = grip
            env.apply_action(g)


def _sample_obj_xy(rng):
    """A randomized, reachable object start spread across the near half of the table (clear of the
    fixed place zone at y=0.15). Wider than the old range so the 50 scenarios are visibly distinct."""
    return (float(rng.uniform(0.42, 0.58)), float(rng.uniform(-0.20, 0.00)))


def _random_start_ee(rng):
    """A randomized reachable EE hover pose above the table -- the per-trial *starting pose* the paper
    permutes (arXiv 2506.09985 4.2), so start frames differ across the 50 scenarios."""
    return np.array([rng.uniform(0.42, 0.58), rng.uniform(-0.16, 0.12), rng.uniform(0.36, 0.48)])


def _far_pose(rng, ref, lo, hi, min_sep=0.12):
    """Draw a reachable pose in the box [lo, hi] that is at least ``min_sep`` from ``ref``."""
    lo = np.asarray(lo); hi = np.asarray(hi)
    p = rng.uniform(lo, hi)
    for _ in range(10):
        if np.linalg.norm(p - ref) >= min_sep:
            break
        p = rng.uniform(lo, hi)
    return p


def _base_arrays(env):
    return {
        "qpos0": env.data.qpos.copy(),
        "qvel0": env.data.qvel.copy(),
        "start_ee": env.get_ee_state().copy(),
        "object_pose": env.object_pose().copy(),
        "zone": env.zone_center().copy(),
    }


# --------------------------------------------------------------------------- per-task builders
def build_reach(env, rng, obj):
    env.reset(cube_xy=_sample_obj_xy(rng))
    start_ee = _random_start_ee(rng)
    _goto(env, start_ee, n=8)  # randomized starting pose (diversity across trials)
    start_img = env.render("planning")
    arrays = _base_arrays(env)
    # target: an independent reachable pose, a large move (>= ~0.18 m) from the start
    ee = env.get_ee_state()[:3]
    target = _far_pose(rng, ee, lo=[0.42, -0.16, 0.33], hi=[0.60, 0.14, 0.46], min_sep=0.18)
    arrays["target"] = target.astype(float)
    arrays["goal_object"] = env.object_pose()[:3].copy()  # object does not move in a reach
    goal_img = env.capture_goal_image(pos=target, euler=EE_DOWN, camera="planning")
    _goto(env, target, n=12)  # validate
    ok = float(np.linalg.norm(env.get_ee_state()[:3] - target)) < 0.02
    return {"start": start_img, "goal": goal_img}, arrays, {"start_grasped": False}, ok


def build_grasp(env, rng, obj):
    env.reset(cube_xy=_sample_obj_xy(rng))
    _goto(env, _random_start_ee(rng), n=8)  # randomized starting pose
    start_img = env.render("planning")
    arrays = _base_arrays(env)
    c = env.object_position()
    dz = OBJECT_SPECS[obj]["grasp_dz"]
    grasp_pos = np.array([c[0], c[1], c[2] + dz])
    arrays["grasp_pos"] = grasp_pos.astype(float)
    goal_img = env.capture_goal_image(pos=grasp_pos, euler=EE_DOWN, gripper=0.0, camera="planning")
    # validate: scripted grasp + lift
    _goto(env, [c[0], c[1], c[2] + 0.14])
    _goto(env, [c[0], c[1], c[2] + dz])
    _goto(env, [c[0], c[1], c[2] + dz], grip=1.0)
    z0 = env.object_position()[2]
    _goto(env, [c[0], c[1], c[2] + 0.16])
    ok = bool(env.gripper_holds_object()) and (env.object_position()[2] - z0) > 0.04
    arrays["goal_object"] = env.object_pose()[:3].copy()
    return {"start": start_img, "goal": goal_img}, arrays, {"start_grasped": False}, ok


def build_reach_with_object(env, rng, obj):
    env.reset(cube_xy=_sample_obj_xy(rng))
    c = env.object_position()
    dz = OBJECT_SPECS[obj]["grasp_dz"]
    # scripted grasp then lift to a start hover above the grasp point -- object STARTS grasped
    _goto(env, [c[0], c[1], c[2] + 0.14])
    _goto(env, [c[0], c[1], c[2] + dz])
    _goto(env, [c[0], c[1], c[2] + dz], grip=1.0)
    start_hover = np.array([c[0], c[1], TABLE_TOP_Z + 0.22])
    _goto(env, start_hover, grip=1.0)
    if not env.gripper_holds_object():
        return None, None, None, False
    start_img = env.render("planning")
    arrays = _base_arrays(env)  # qpos0 = grasped, object-in-hand state
    ee = env.get_ee_state()[:3]
    # goal: a large, paper-like traverse across the table while holding (>= ~0.20 m)
    goal_ee = _far_pose(rng, ee, lo=[0.40, 0.04, 0.30], hi=[0.60, 0.24, 0.46], min_sep=0.20)
    arrays["goal_ee"] = goal_ee.astype(float)
    goal_img = env.capture_goal_image(pos=goal_ee, euler=EE_DOWN, gripper=1.0, camera="planning",
                                      held_object=True)
    _goto(env, goal_ee, grip=1.0, n=10)  # validate the held traverse
    ok = bool(env.gripper_holds_object())
    arrays["goal_object"] = env.object_pose()[:3].copy()
    return {"start": start_img, "goal": goal_img}, arrays, {"start_grasped": True}, ok


def build_pick_place(env, rng, obj):
    env.reset(cube_xy=_sample_obj_xy(rng))
    _goto(env, _random_start_ee(rng), n=8)  # randomized starting pose
    start_img = env.render("planning")
    arrays = _base_arrays(env)  # qpos0 = initial, object on the table
    c = env.object_position()
    dz = OBJECT_SPECS[obj]["grasp_dz"]
    half = OBJECT_SPECS[obj]["rest_half_z"]
    zone = env.zone_center()
    grasp_pos = np.array([c[0], c[1], c[2] + dz])
    grasped_pos = np.array([c[0], c[1], c[2] + dz + 0.04])  # grasped + slightly lifted (sub-goal 1)
    vicinity_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + half + 0.12])
    place_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + half + 0.04])
    arrays["grasp_pos"] = grasp_pos.astype(float)
    arrays["vicinity_pos"] = vicinity_pos.astype(float)
    arrays["place_pos"] = place_pos.astype(float)
    # scripted grasp FIRST, so all three sub-goals show the SAME grasp (object held), differing only
    # in location -- sub-goal 1 = grasped at the object, 2 = held in the vicinity, 3 = placed.
    _goto(env, [c[0], c[1], c[2] + 0.14])
    _goto(env, [c[0], c[1], c[2] + dz])
    _goto(env, [c[0], c[1], c[2] + dz], grip=1.0)
    _goto(env, grasped_pos, grip=1.0)
    if not env.gripper_holds_object():
        return None, None, None, False
    goal_1 = env.capture_goal_image(pos=grasped_pos, euler=EE_DOWN, gripper=1.0, camera="planning",
                                    held_object=True)   # cup grasped (rim grip) at its location
    goal_2 = env.capture_goal_image(pos=vicinity_pos, euler=EE_DOWN, gripper=1.0, camera="planning",
                                    held_object=True)
    goal_g = env.capture_goal_image(pos=place_pos, euler=EE_DOWN, gripper=1.0, camera="planning",
                                    held_object=True)
    # validate the full place
    _goto(env, vicinity_pos, grip=1.0)
    _goto(env, place_pos, grip=1.0)
    _goto(env, place_pos, grip=-1.0)
    for _ in range(2):
        _goto(env, place_pos + np.array([0, 0, 0.10]), n=3)
    err = float(np.linalg.norm(env.object_position()[:2] - zone[:2]))
    ok = err < 0.06
    arrays["goal_object"] = np.array([zone[0], zone[1], TABLE_TOP_Z + half])
    return {"start": start_img, "goal_1": goal_1, "goal_2": goal_2, "goal": goal_g}, arrays, \
        {"start_grasped": False}, ok


BUILDERS = {
    "reach": build_reach,
    "grasp": build_grasp,
    "reach_with_object": build_reach_with_object,
    "pick_place": build_pick_place,
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
    return {"reach": 0, "grasp": 1, "reach_with_object": 2, "pick_place": 3}[task]


if __name__ == "__main__":
    main()
