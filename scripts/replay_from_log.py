"""Reproduce a benchmark trial in MuJoCo purely from the LOG -- no V-JEPA, no GPU, deterministic.

The insight: the benchmark scenario is a FIXED bundle (deterministic start ``qpos0``), and steps.csv
logs V-JEPA's planned action (dx, dy, dz, dgrip) at every planning step. MuJoCo physics is
deterministic, so restoring the bundle start and re-applying the logged actions reproduces the exact
rollout -- including the scripted close/lift/place tail (pure geometric primitives). This is more
faithful than re-running V-JEPA (whose bf16 CEM is nondeterministic).

Produces a per-step qpos npz that ``scripts/replay_rollout_viewer.py`` opens for N/B 3D stepping.

    # reproduce grasp/cup trial 44 from the latest run and open the 3D viewer
    python scripts/replay_from_log.py --task grasp --object cup --trial 44 --view

    # or point at a specific run and just write the npz
    python scripts/replay_from_log.py --run logs/closed_loop_runs/<id> --task pick_place --object box --trial 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# scripted settle length per task (matches run_bundle_trial); other primitives are task-independent
_SETTLE = {"grasp": 20, "reach_with_object": 10, "grasp_and_reach": 10,
           "pick_place": 30, "place_with_object": 30}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _latest_run():
    runs = sorted(glob.glob(os.path.join(_REPO_ROOT, "logs", "closed_loop_runs", "*")))
    return runs[-1] if runs else None


def _scripted(env, cap, phase, target=None, gripper=None, dz=None, settle=0, n=4):
    """Re-run one scripted primitive (matches run_closed_loop_benchmark.scripted), capturing qpos
    after every apply_action into ``cap`` as (phase, qpos, grip_cmd)."""
    if target is not None:
        for _ in range(n):
            cur = env.get_ee_state()[:3]
            d = np.zeros(7)
            d[:3] = np.asarray(target) - cur
            env.apply_action(d)
            cap(phase, gripper)
    if gripper is not None:
        for _ in range(3):
            g = np.zeros(7)
            g[6] = gripper
            env.apply_action(g)
            cap(phase, gripper)
    if dz is not None:
        for _ in range(n):
            d = np.zeros(7)
            d[2] = dz / n
            env.apply_action(d)
            cap(phase, gripper)
    if settle:
        hold = env.get_ee_state()[:3].copy()
        for _ in range(settle):
            cur = env.get_ee_state()[:3]
            d = np.zeros(7)
            d[:3] = hold - cur
            env.apply_action(d)
            cap(phase, gripper)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run dir (default: latest under logs/closed_loop_runs)")
    p.add_argument("--task", required=True)
    p.add_argument("--object", required=True, choices=["cup", "box"])
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--bundles", default="tasks")
    p.add_argument("--out", default=None, help="output npz (default: logs/rollouts/<run>/<label>.npz)")
    p.add_argument("--view", action="store_true", help="open the 3D viewer on the reproduced rollout")
    args = p.parse_args()

    run_dir = args.run or _latest_run()
    if not run_dir:
        raise SystemExit("no run dir found")
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(_REPO_ROOT, run_dir)
    label = f"{args.task}/{args.object}"

    cfg = {}
    if os.path.exists(os.path.join(run_dir, "run_config.json")):
        with open(os.path.join(run_dir, "run_config.json")) as fh:
            cfg = json.load(fh)
    cam_preset = (cfg.get("bundles", {}) or {}).get("planning_camera")

    # bundle_id for this trial (from trials.csv)
    bundle_id = None
    with open(os.path.join(run_dir, "trials.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["task"] == label and int(r["trial"]) == args.trial:
                bundle_id = r["bundle_id"]
                break
    if not bundle_id:
        raise SystemExit(f"trial {label} #{args.trial} not found in {run_dir}/trials.csv")

    # logged per-step rows for this trial (in order)
    steps = []
    with open(os.path.join(run_dir, "steps.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["task"] == label and int(r["trial"]) == args.trial:
                steps.append(r)

    # the bundle: deterministic start + zone + meta
    bdir = os.path.join(_REPO_ROOT, args.bundles, args.task, args.object, bundle_id)
    arr = dict(np.load(os.path.join(bdir, "arrays.npz")))
    with open(os.path.join(bdir, "meta.json")) as fh:
        meta = json.load(fh)
    start_grasped = bool(meta.get("start_grasped", False))

    import importlib
    fb = importlib.import_module("src.envs.franka_build")
    from src.envs.franka_droid_env import FrankaDroidEnv

    cam_override = None
    presets = {"A_current": {}, "B_closer": {"distance": 1.05},
               "C_droidlike": {"azimuth": -135.0, "elevation": -30.0, "distance": 1.3}}
    if cam_preset in presets:
        cam_override = presets[cam_preset]

    env = FrankaDroidEnv(add_object=True, add_zone=True, object_type=args.object,
                         add_distractors=True, planning_camera=cam_override)
    frames, phases, grips, held, tilt = [], [], [], [], []

    def cap(phase, grip_cmd):
        frames.append(env.data.qpos.copy())
        phases.append(phase)
        grips.append(float(grip_cmd) if grip_cmd is not None else 0.0)
        held.append(1.0 if env.gripper_holds_object() else 0.0)
        tilt.append(float(np.degrees(env.object_tilt())))

    # restore the exact recorded start
    env.reset()
    zone = arr.get("zone")
    if zone is not None and np.all(np.isfinite(zone)):
        env.set_zone_xy(float(zone[0]), float(zone[1]))
    env.set_state(arr["qpos0"], gripper=(1.0 if start_grasped else 0.0),
                  settle=(8 if start_grasped else 0))
    cap("start", 1.0 if start_grasped else 0.0)

    settle_n = _SETTLE.get(args.task, 20)
    for row in steps:
        phase = row["phase"]
        if phase.startswith("vjepa"):
            a = np.array([_f(row["dx"]), _f(row["dy"]), _f(row["dz"]), 0.0, 0.0, 0.0, _f(row["dgrip"])])
            if np.any(~np.isfinite(a)):
                continue
            env.apply_action(a)
            cap(phase, a[6])
        elif phase == "close":
            _scripted(env, cap, phase, gripper=1.0)
        elif phase == "open":
            _scripted(env, cap, phase, gripper=-1.0)
        elif phase == "lift":
            _scripted(env, cap, phase, dz=0.12)
        elif phase == "lower":
            cur = env.get_ee_state()[:3]
            _scripted(env, cap, phase, target=[cur[0], cur[1], fb.TABLE_TOP_Z + 0.05])
        elif phase == "settle":
            _scripted(env, cap, phase, settle=settle_n)
        # 'final' etc.: no action, already captured

    env.close()

    out = args.out or os.path.join(_REPO_ROOT, "logs", "rollouts",
                                   os.path.basename(run_dir),
                                   f"{args.task}_{args.object}_{bundle_id}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out,
             qpos=np.asarray(frames, dtype=float),
             phase=np.asarray(phases),
             dist=np.full(len(frames), np.nan),
             energy=np.full(len(frames), np.nan),
             tilt=np.asarray(tilt, dtype=float),
             held=np.asarray(held, dtype=float),
             grip=np.asarray(grips, dtype=float),
             object_type=args.object, task=args.task, trial=int(args.trial))
    print(f"reproduced {label} #{args.trial} ({bundle_id}) -> {os.path.relpath(out, _REPO_ROOT)} "
          f"({len(frames)} frames)")

    if args.view:
        viewer = os.path.join(_REPO_ROOT, "scripts", "replay_rollout_viewer.py")
        subprocess.run([sys.executable, viewer, "--object", args.object,
                        "--dir", os.path.dirname(out), "--label", f"{args.task}#{args.trial}"])


if __name__ == "__main__":
    main()
