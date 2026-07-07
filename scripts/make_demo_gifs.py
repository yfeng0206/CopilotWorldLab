"""Render labeled demo GIFs of real benchmark rollouts, reproduced from the LOG (no GPU model).

For each requested (task, object, trial) it restores the fixed bundle start and re-applies V-JEPA's
logged per-step actions + the deterministic scripted tail (identical to replay_from_log), rendering
each step from the run's planning camera and overlaying a label (task/object, HIT/MISS, final error,
per-frame phase/held). Writes short GIFs suitable for a demo reel.

Reproduction is deterministic MuJoCo physics -- it does NOT run V-JEPA, so it is safe to run while a
benchmark occupies the GPU model.

    python scripts/make_demo_gifs.py            # auto: 1 HIT + 1 MISS for the 4 completed groups
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from replay_from_log import _f, _scripted, _SETTLE  # noqa: E402

_PRESETS = {"A_current": {}, "B_closer": {"distance": 1.05},
            "C_droidlike": {"azimuth": -135.0, "elevation": -30.0, "distance": 1.3}}


def _run_tag(d):
    p = os.path.join(d, "run_config.json")
    if not os.path.exists(p):
        return ""
    rep = (json.load(open(p)).get("logs", {}) or {}).get("report_dir", "")
    for seg in rep.replace("\\", "/").split("/"):
        if seg.startswith("closed_loop_"):
            return seg[len("closed_loop_"):]
    return ""


def _find_group(task, obj, tag="full800_B"):
    """Return (run_dir, rows) for the (task,object) group from the run with the most trials."""
    lab = f"{task}/{obj}"
    best_dir, best_rows = None, []
    for d in sorted(glob.glob(os.path.join(_REPO_ROOT, "logs", "closed_loop_runs", "*"))):
        if _run_tag(d) != tag:
            continue
        tc = os.path.join(d, "trials.csv")
        if not os.path.exists(tc):
            continue
        rows = [r for r in csv.DictReader(open(tc)) if r["task"] == lab]
        if len(rows) > len(best_rows):
            best_dir, best_rows = d, rows
    return best_dir, best_rows


def _pick(rows, want_hit):
    pool = [r for r in rows if r["success_loose"] == ("1" if want_hit else "0")]
    if not pool:
        return None
    return sorted(pool, key=lambda r: float(r["error_m"]))[0]


def _label_frame(rgb, banner, caption, ok):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    try:
        f_big = ImageFont.truetype("arialbd.ttf", 22)
        f_small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        f_big = ImageFont.load_default()
        f_small = ImageFont.load_default()
    bar = (30, 130, 40) if ok else (150, 40, 40)          # green hit / red miss
    d.rectangle([0, 0, w, 34], fill=bar)
    d.text((8, 6), banner, fill=(255, 255, 255), font=f_big)
    d.rectangle([0, h - 26, w, h], fill=(0, 0, 0))
    d.text((8, h - 23), caption, fill=(230, 230, 230), font=f_small)
    return np.asarray(img)


def render_demo(run_dir, task, obj, trial, outcome, final_err_cm, failure, out_gif, hw=480):
    import imageio.v2 as imageio

    from src.envs.franka_build import TABLE_TOP_Z
    from src.envs.franka_droid_env import FrankaDroidEnv

    lab = f"{task}/{obj}"
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    cam_preset = (cfg.get("bundles", {}) or {}).get("planning_camera")
    cam_override = _PRESETS.get(cam_preset)

    bundle_id = next(r["bundle_id"] for r in csv.DictReader(open(os.path.join(run_dir, "trials.csv")))
                     if r["task"] == lab and int(r["trial"]) == trial)
    steps = [r for r in csv.DictReader(open(os.path.join(run_dir, "steps.csv")))
             if r["task"] == lab and int(r["trial"]) == trial]

    bdir = os.path.join(_REPO_ROOT, "tasks", task, obj, bundle_id)
    arr = dict(np.load(os.path.join(bdir, "arrays.npz")))
    meta = json.load(open(os.path.join(bdir, "meta.json")))
    start_grasped = bool(meta.get("start_grasped", False))

    env = FrankaDroidEnv(render_width=hw, render_height=hw, add_object=True, add_zone=True,
                         object_type=obj, add_distractors=True, planning_camera=cam_override)
    ok = outcome == "HIT"
    banner = f"{lab}  -  {outcome}"
    frames = []

    def cap(phase, grip_cmd):
        rgb = env.render(camera="planning")
        held = "yes" if env.gripper_holds_object() else "no"
        tilt = np.degrees(env.object_tilt())
        cap_txt = f"{phase}  |  held={held}  tilt={tilt:.0f}deg  |  final err={final_err_cm:.1f}cm"
        if not ok and failure:
            cap_txt += f"  ({failure})"
        frames.append(_label_frame(rgb, banner, cap_txt, ok))

    env.reset()
    zone = arr.get("zone")
    if zone is not None and np.all(np.isfinite(zone)):
        env.set_zone_xy(float(zone[0]), float(zone[1]))
    env.set_state(arr["qpos0"], gripper=(1.0 if start_grasped else 0.0),
                  settle=(8 if start_grasped else 0))
    cap("start", 1.0 if start_grasped else 0.0)

    settle_n = _SETTLE.get(task, 20)
    for row in steps:
        phase = row["phase"]
        if phase.startswith("vjepa"):
            a = np.array([_f(row["dx"]), _f(row["dy"]), _f(row["dz"]), 0., 0., 0., _f(row["dgrip"])])
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
            _scripted(env, cap, phase, target=[cur[0], cur[1], TABLE_TOP_Z + 0.05])
        elif phase == "settle":
            _scripted(env, cap, phase, settle=settle_n)
    env.close()

    # hold the last frame a beat so the outcome is readable
    frames += [frames[-1]] * 6
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, frames, fps=10, loop=0)
    print(f"  {lab:24s} {outcome:4s} trial {trial:>2} -> {os.path.relpath(out_gif, _REPO_ROOT)} "
          f"({len(frames)} frames)")


def main():
    groups = [("grasp", "cup"), ("grasp", "box"),
              ("reach_with_object", "cup"), ("reach_with_object", "box")]
    out_dir = os.path.join(_REPO_ROOT, "results", "demos", "full800_B")
    for task, obj in groups:
        run_dir, rows = _find_group(task, obj)
        if not rows:
            print(f"  {task}/{obj}: no data yet, skipping")
            continue
        for want_hit, outcome in [(True, "HIT"), (False, "MISS")]:
            r = _pick(rows, want_hit)
            if not r:
                print(f"  {task}/{obj}: no {outcome} example, skipping")
                continue
            out = os.path.join(out_dir, f"{task}_{obj}_{outcome}.gif")
            render_demo(run_dir, task, obj, int(r["trial"]), outcome,
                        float(r["error_m"]) * 100, r["failure"], out)


if __name__ == "__main__":
    main()
