"""Interactive MuJoCo viewer to inspect the fixed task bundles stage by stage.

Opens a live MuJoCo window for one object (cup or box). All four tasks (grasp, reach_with_object,
grasp_and_reach, pick_place) share the same scene for a given object, so this cycles through every
task's stages (start -> sub-goals -> goal) in one window. Orbit/zoom freely with the mouse; press
SPACE to advance to the next stage, BACKSPACE to go back. The stage is frozen (physics not stepped),
so you see exactly the saved state.

    python scripts/inspect_task_viewer.py --object cup --tasks-dir tasks_verify
    python scripts/inspect_task_viewer.py --object box --tasks-dir tasks_verify

Requires per-stage qpos in the bundles (arrays.npz: qpos_start / qpos_goal_1 / qpos_goal_2 /
qpos_goal), written by scripts/generate_task_bundles.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TASK_ORDER = ["grasp", "reach_with_object", "grasp_and_reach", "pick_place"]
STAGE_KEYS = ["qpos_start", "qpos_goal_1", "qpos_goal_2", "qpos_goal"]
KEY_SPACE = 32
KEY_BACKSPACE = 259  # GLFW


def _collect_stages(tasks_dir, obj):
    """Return [(label, qpos), ...] over all tasks (first bundle each) for this object."""
    stages = []
    for task in TASK_ORDER:
        obj_dir = os.path.join(tasks_dir, task, obj)
        if not os.path.isdir(obj_dir):
            continue
        ids = sorted(os.listdir(obj_dir))
        if not ids:
            continue
        arrays = np.load(os.path.join(obj_dir, ids[0], "arrays.npz"))
        for key in STAGE_KEYS:
            if key in arrays.files:
                stage = key.replace("qpos_", "")
                stages.append((f"{task}  [{stage}]  ({ids[0]})", arrays[key].copy()))
    return stages


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", choices=["cup", "box"], default="cup")
    p.add_argument("--tasks-dir", default=os.path.join(_REPO_ROOT, "tasks_verify"))
    args = p.parse_args()

    import mujoco
    import mujoco.viewer
    from src.envs.franka_build import build_franka_robotiq

    model = build_franka_robotiq(add_object=True, add_zone=True, object_type=args.object,
                                 add_distractors=True)
    data = mujoco.MjData(model)

    stages = _collect_stages(args.tasks_dir, args.object)
    if not stages:
        raise SystemExit(f"no stages found under {args.tasks_dir} for object={args.object} "
                         f"(generate bundles first)")

    idx = {"i": 0}

    def apply(i):
        label, qpos = stages[i]
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        print(f"[{i + 1}/{len(stages)}]  {label}", flush=True)

    def key_callback(keycode):
        if keycode == KEY_SPACE:
            idx["i"] = (idx["i"] + 1) % len(stages)
            apply(idx["i"])
        elif keycode == KEY_BACKSPACE:
            idx["i"] = (idx["i"] - 1) % len(stages)
            apply(idx["i"])

    print(f"=== inspecting object={args.object}: {len(stages)} stages ===")
    print("SPACE = next stage, BACKSPACE = previous stage, mouse = orbit/zoom, close window = quit")
    apply(0)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)   # hold the stage frozen (no physics step)
            viewer.sync()
            time.sleep(0.03)


if __name__ == "__main__":
    main()
