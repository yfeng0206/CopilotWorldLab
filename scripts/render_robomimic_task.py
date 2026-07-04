"""Stage 0 (closed-loop success benchmark): render robomimic raw demos into task bundles.

Reads a robomimic ``demo_v15.hdf5``, renders each selected demo's raw states to images on Windows
(plain MuJoCo + patched robosuite assets, no robosuite runtime), and writes a task *bundle* per
demo (start/goal images + privileged states + patched model XML; see src/bench/schema.py). Also
writes a **contact sheet** and **rollout GIF** for the human visual check the benchmark plan
requires *before* any benchmarking (docs/experiments/closed_loop_success_plan.md).

    python scripts/render_robomimic_task.py --task lift --demos 3
    # -> tasks/lift_ph_demo0/ ... and results/benchmarks/closed_loop/lift_ph_demo0_contact.png
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.bench.schema import SUCCESS_DEFAULTS, TaskBundle  # noqa: E402
from src.envs.robomimic_render import RobomimicDemoRenderer  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("render_robomimic")

# robomimic task -> (our task_type, canonical dataset relpath). Lift is a grasp-lift; Can/Square
# are pick-and-place -> place. Success thresholds come from SUCCESS_DEFAULTS.
TASKS = {
    "lift": ("grasp_lift", "v1.5/lift/ph/demo_v15.hdf5"),
    "can": ("place", "v1.5/can/ph/demo_v15.hdf5"),
    "square": ("place", "v1.5/square/ph/demo_v15.hdf5"),
}


def _annotate(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def contact_sheet(rndr, cams, frames, actions, ee_by_frame, out_png, title):
    n = len(frames)
    cols = 4
    grid_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(1 + grid_rows, cols, figsize=(cols * 2.6, (1 + grid_rows) * 2.6))
    axes = np.atleast_2d(axes)
    # header row: start/goal at two cameras
    cam1 = cams[0]
    cam2 = cams[1] if len(cams) > 1 else cams[0]
    rndr.set_frame(0)
    start1 = rndr.render(cam1)
    start2 = rndr.render(cam2)
    rndr.set_frame(rndr.num_frames - 1)
    goal1 = rndr.render(cam1)
    goal2 = rndr.render(cam2)
    _annotate(axes[0, 0], start1, f"START [{cam1}]")
    _annotate(axes[0, 1], goal1, f"GOAL [{cam1}]")
    _annotate(axes[0, 2], start2, f"START [{cam2}]")
    _annotate(axes[0, 3], goal2, f"GOAL [{cam2}]")
    # trajectory grid at cam1 with state/action overlay
    for k, t in enumerate(frames):
        r, c = 1 + k // cols, k % cols
        img = ee_by_frame[t][1]
        ee = ee_by_frame[t][0]
        a = actions[t] if t < len(actions) else np.zeros(7)
        cap = (f"f{t}  ee=({ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}) g={ee[6]:.2f}\n"
               f"a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}|{a[6]:+.0f})")
        _annotate(axes[r, c], img, cap)
    for k in range(n, grid_rows * cols):
        axes[1 + k // cols, k % cols].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(TASKS), default="lift")
    parser.add_argument("--dataset", default=None, help="override path to demo_v15.hdf5")
    parser.add_argument("--demos", type=int, default=3, help="number of demos (from demo_0)")
    parser.add_argument("--camera", default="agentview", help="planning camera for start/goal")
    parser.add_argument("--alt-camera", default="frontview", help="2nd camera in the contact sheet")
    parser.add_argument("--frames", type=int, default=8, help="trajectory frames in the contact sheet")
    parser.add_argument("--gif-frames", type=int, default=40)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--tasks-dir", default=os.path.join(_REPO_ROOT, "tasks"))
    args = parser.parse_args()

    task_type, relpath = TASKS[args.task]
    dataset = args.dataset or os.path.join(_REPO_ROOT, "data", "robomimic", relpath)
    if not os.path.exists(dataset):
        logger.error("missing dataset %s (download the robomimic raw demo first)", dataset)
        raise SystemExit(1)

    prov_dir = os.path.join(_REPO_ROOT, "results", "benchmarks", "closed_loop")
    os.makedirs(prov_dir, exist_ok=True)

    with RobomimicDemoRenderer(dataset, height=args.height, width=args.width) as rndr:
        cams = [args.camera, args.alt_camera]
        logger.info("dataset=%s planning=%s alt=%s", os.path.basename(dataset),
                    args.camera, args.alt_camera)
        for di, demo_name in enumerate(rndr.demo_names[: args.demos]):
            rndr.load_demo(demo_name)
            T = rndr.num_frames
            if di == 0:
                logger.info("model cameras=%s | object=%s eef=%s",
                            rndr.cameras, rndr.object_body, rndr.eef_site)
            for cam in cams:
                if cam not in rndr.cameras:
                    logger.error("camera %s not in model %s", cam, rndr.cameras)
                    raise SystemExit(1)

            # per-frame EE state + planning-camera render (cached for the contact sheet/GIF)
            ee_by_frame = {}
            traj_imgs = []
            for t in range(T):
                rndr.set_frame(t)
                ee_by_frame[t] = (rndr.ee_state(), rndr.render(args.camera))
                traj_imgs.append(ee_by_frame[t][1])

            rndr.set_frame(0)
            start_img = rndr.render(args.camera)
            start_state, obj0 = rndr.ee_state(), rndr.object_state()
            qpos0, qvel0 = rndr.flattened_state(0)
            rndr.set_frame(T - 1)
            goal_img = rndr.render(args.camera)
            goal_state, objg = rndr.ee_state(), rndr.object_state()

            task_id = f"{args.task}_ph_demo{di}"
            meta = {
                "task_id": task_id, "task_type": task_type, "difficulty": "demo_default",
                "source": f"robomimic/{args.task}/ph/{demo_name}", "camera": args.camera,
                "alt_camera": args.alt_camera, "image_hw": [args.height, args.width],
                "fps": args.fps, "units": "meters",
                "robot_ee_convention": "xyz+euler(extrinsic XYZ)+gripper (7-D)",
                "object_body": rndr.object_body, "eef_site": rndr.eef_site,
                "target": None, "success_spec": {"type": task_type, **SUCCESS_DEFAULTS[task_type]},
                "n_frames": T, "seed": 0,
                "note": "goal = demo final frame; multi-stage goals TBD for place tasks",
            }
            bundle = TaskBundle(
                meta=meta,
                images={"start": start_img, "goal": goal_img},
                arrays={
                    "start_state": start_state, "goal_state": goal_state,
                    "object_state": obj0, "goal_object_state": objg,
                    "qpos0": qpos0, "qvel0": qvel0, "actions": rndr.actions,
                },
                model_xml=rndr.model_xml,
            )
            out = bundle.save(args.tasks_dir)

            frames = np.linspace(0, T - 1, min(args.frames, T)).astype(int).tolist()
            cs_path = os.path.join(out, "contact_sheet.png")
            contact_sheet(rndr, cams, frames, rndr.actions, ee_by_frame, cs_path,
                          f"{task_id}  ({task_type}, {os.path.basename(dataset)}, {demo_name}, T={T})")
            shutil.copy(cs_path, os.path.join(prov_dir, f"{task_id}_contact.png"))

            gif_idx = np.linspace(0, T - 1, min(args.gif_frames, T)).astype(int)
            imageio.mimsave(os.path.join(out, "rollout.gif"),
                            [traj_imgs[i] for i in gif_idx], fps=args.fps, loop=0)

            logger.info("%-16s T=%d obj=%s -> %s", task_id, T, rndr.object_body, out)
            logger.info("  start_ee=%s", np.array2string(start_state, precision=3))
            logger.info("  goal_ee =%s", np.array2string(goal_state, precision=3))
            logger.info("  obj dz (goal-start) = %+.3f m", float(objg[2] - obj0[2]))

    logger.info("done -- review contact sheets in %s, then approve before benchmarking", prov_dir)


if __name__ == "__main__":
    main()
