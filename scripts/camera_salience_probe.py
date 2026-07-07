"""Static camera-salience probe for the grasp task (no model, no CEM).

Renders the GRASPED goal state of a grasp bundle from candidate planning cameras and measures how
visible the contact is at 256x256 -- object pixel area, gripper pixel area, and the combined
object+gripper bounding-box size -- using MuJoCo segmentation. A camera that shows more object/
gripper pixels and a tighter, clearer contact gives the frozen encoder a stronger signal to grasp
precisely. Also saves an RGB contact sheet (one column per camera) per object for eyeballing.

This is the cheap pre-check before any GPU benchmark: if a candidate camera does not increase
object/gripper salience over the current view, it is not worth a full V-JEPA run.

    python scripts/camera_salience_probe.py --objects cup box

Cameras (edit CAMERAS below): A=current az45_el45; B=closer same-angle; C=DROID-like left-exo.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Candidate cameras as partial overrides of PLANNING_CAMERA (az -45 / el -45 / dist 1.5 / lookat).
CAMERAS = {
    "A_current":   {},                                                    # baseline az45_el45
    "B_closer":    {"distance": 1.05},                                    # same angle, tighter
    "C_droidlike": {"azimuth": -135.0, "elevation": -30.0, "distance": 1.3},  # left exocentric
}


def _seg_counts(model, data, mujoco, cam, hw, cube_bid, is_gripper_body):
    """Render segmentation from ``cam`` and return (obj_px, grip_px, bbox_wh, rgb)."""
    seg_r = mujoco.Renderer(model, height=hw, width=hw)
    seg_r.enable_segmentation_rendering()
    seg_r.update_scene(data, camera=cam)
    seg = seg_r.render()[:, :, 0]            # geom id per pixel (-1 = background)
    seg_r.close()

    rgb_r = mujoco.Renderer(model, height=hw, width=hw)
    rgb_r.update_scene(data, camera=cam)
    rgb = rgb_r.render()
    rgb_r.close()

    obj_mask = np.zeros(seg.shape, dtype=bool)
    grip_mask = np.zeros(seg.shape, dtype=bool)
    for gid in np.unique(seg):
        if gid < 0:
            continue
        bid = int(model.geom_bodyid[gid])
        px = seg == gid
        if bid == cube_bid:
            obj_mask |= px
        elif is_gripper_body(bid):
            grip_mask |= px

    both = obj_mask | grip_mask
    if both.any():
        ys, xs = np.where(both)
        bbox_wh = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    else:
        bbox_wh = (0, 0)
    return int(obj_mask.sum()), int(grip_mask.sum()), bbox_wh, rgb


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", nargs="+", default=["cup", "box"], choices=["cup", "box"])
    p.add_argument("--hw", type=int, default=256, help="render resolution (matches the benchmark)")
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "results", "camera_salience"))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco

    from src.envs.franka_build import make_free_camera
    from src.envs.franka_droid_env import FrankaDroidEnv

    os.makedirs(args.out, exist_ok=True)
    total_px = float(args.hw * args.hw)
    rows = []

    for obj in args.objects:
        bundles = sorted(glob.glob(os.path.join(_REPO_ROOT, "tasks", "grasp", obj, "*", "arrays.npz")))
        if not bundles:
            print(f"no grasp/{obj} bundles found; skipping")
            continue
        qpos_goal = np.load(bundles[0])["qpos_goal"]   # grasped/contact state

        env = FrankaDroidEnv(render_width=args.hw, render_height=args.hw,
                             add_object=True, add_zone=True, object_type=obj, add_distractors=True)
        env.set_state(qpos_goal)
        model, data = env.model, env.data
        cube_bid = env._cube_bid
        is_grip = env._is_gripper_body

        fig, axes = plt.subplots(1, len(CAMERAS), figsize=(4 * len(CAMERAS), 4))
        if len(CAMERAS) == 1:
            axes = [axes]
        for ax, (name, override) in zip(axes, CAMERAS.items()):
            # overrides are relative to the validated PLANNING_CAMERA default
            from src.envs.franka_build import PLANNING_CAMERA
            spec = {**PLANNING_CAMERA, **override}
            cam = make_free_camera(**spec)
            obj_px, grip_px, bbox, rgb = _seg_counts(model, data, mujoco, cam, args.hw,
                                                     cube_bid, is_grip)
            rows.append({"object": obj, "camera": name,
                         "obj_pct": 100 * obj_px / total_px, "grip_pct": 100 * grip_px / total_px,
                         "bbox": f"{bbox[0]}x{bbox[1]}"})
            ax.imshow(rgb)
            ax.set_title(f"{name}\nobj={100 * obj_px / total_px:.2f}%  "
                         f"grip={100 * grip_px / total_px:.2f}%\nbbox={bbox[0]}x{bbox[1]}",
                         fontsize=9)
            ax.axis("off")
        env.close()
        sheet = os.path.join(args.out, f"grasp_{obj}_camera_salience.png")
        fig.suptitle(f"grasp / {obj}: grasped-goal salience per camera (higher obj/grip % = better)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(sheet, dpi=110)
        plt.close(fig)
        print(f"saved {os.path.relpath(sheet, _REPO_ROOT)}")

    print("\n=== camera salience (grasped goal state, % of 256x256 frame) ===")
    print(f"{'object':6s} {'camera':12s} {'obj_px%':>8s} {'grip_px%':>9s} {'bbox':>10s}")
    for r in rows:
        print(f"{r['object']:6s} {r['camera']:12s} {r['obj_pct']:8.2f} {r['grip_pct']:9.2f} "
              f"{r['bbox']:>10s}")


if __name__ == "__main__":
    main()
