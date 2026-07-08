"""Side-by-side demo: the SAME logged rollout under the stiff position servo (as benchmarked) vs a
force-limited (compliant) arm.

The benchmark arm is a stiff position servo: it drives the IK joint target with up to the real Franka
torque (+/-87 Nm), which is plenty to shove a ~30 g object through a soft table contact. Capping that
force makes the servo STALL at contact instead of bulldozing -- a passive robustness knob, not a
force-feedback controller. Free-space reaching is unchanged (forces stay far below the cap), so this
isolates the contact behavior.

Replay is deterministic MuJoCo physics from the log (no V-JEPA, no GPU), so it is safe to run while a
benchmark occupies the GPU.

    python scripts/make_compliance_demo.py --task grasp --object box --trial 41 --forcelim 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

from replay_from_log import _f, _scripted, _SETTLE            # noqa: E402
from rescore_from_log import _PRESETS, _THR, _apply_mods, _find_group   # noqa: E402


def _outcome(env, task, arr, grasp_error, obj_z0):
    obj = env.object_position()
    tilt = float(np.degrees(env.object_tilt()))
    held = bool(env.gripper_holds_object())
    speed = float(env.object_speed())
    if task == "grasp":
        err = grasp_error if grasp_error is not None else 1.0
        lifted = (obj[2] - (obj_z0 if obj_z0 is not None else obj[2])) > 0.04
        ok = err < _THR[task] and lifted and held and tilt < 30.0 and speed < 0.05
    elif task in ("reach_with_object", "grasp_and_reach"):
        err = float(np.linalg.norm(obj - np.asarray(arr["goal_object"], dtype=float)))
        ok = err < _THR[task] and held and tilt < 30.0
    else:
        err = float(np.linalg.norm(obj[:2] - env.zone_center()))
        ok = err < _THR[task] and tilt < 25.0 and speed < 0.05 and bool(env.object_placed())
    return ok, held, tilt, err


def _panel(rgb, title, title_bg, sub):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    try:
        f_t = ImageFont.truetype("arialbd.ttf", 18)
        f_s = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        f_t = ImageFont.load_default()
        f_s = ImageFont.load_default()
    d.rectangle([0, 0, w, 28], fill=title_bg)
    d.text((7, 5), title, fill=(255, 255, 255), font=f_t)
    d.rectangle([0, h - 22, w, h], fill=(0, 0, 0))
    d.text((7, h - 19), sub, fill=(235, 235, 235), font=f_s)
    return np.asarray(img)


def _build(obj, cam_override, hw):
    from src.envs.franka_droid_env import FrankaDroidEnv
    return FrankaDroidEnv(render_width=hw, render_height=hw, add_object=True, add_zone=True,
                          object_type=obj, add_distractors=True, planning_camera=cam_override)


def render_pair(run_dir, task, obj, trial, mods, out_gif, hw=360):
    import imageio.v2 as imageio
    from src.envs.franka_build import TABLE_TOP_Z

    lab = f"{task}/{obj}"
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    cam_override = _PRESETS.get((cfg.get("bundles", {}) or {}).get("planning_camera"))
    bundle_id = next(r["bundle_id"] for r in csv.DictReader(open(os.path.join(run_dir, "trials.csv")))
                     if r["task"] == lab and int(r["trial"]) == trial)
    steps = [r for r in csv.DictReader(open(os.path.join(run_dir, "steps.csv")))
             if r["task"] == lab and int(r["trial"]) == trial]
    bdir = os.path.join(_REPO_ROOT, "tasks", task, obj, bundle_id)
    arr = dict(np.load(os.path.join(bdir, "arrays.npz")))
    meta = json.load(open(os.path.join(bdir, "meta.json")))
    start_grasped = bool(meta.get("start_grasped", False))

    envs = {"stiff": _build(obj, cam_override, hw), "soft": _build(obj, cam_override, hw)}
    _apply_mods(envs["soft"], mods)
    grasp_error = {"stiff": None, "soft": None}
    obj_z0 = {"stiff": None, "soft": None}
    saw_scripted = {"stiff": False, "soft": False}
    frames = {"stiff": [], "soft": []}

    for key, env in envs.items():
        env.reset()
        zone = arr.get("zone")
        if zone is not None and np.all(np.isfinite(zone)):
            env.set_zone_xy(float(zone[0]), float(zone[1]))
        env.set_state(arr["qpos0"], gripper=(1.0 if start_grasped else 0.0),
                      settle=(8 if start_grasped else 0))

    settle_n = _SETTLE.get(task, 20)
    title = {"stiff": "STIFF SERVO  (as benchmarked)", "soft": "FORCE-LIMITED  (compliant)"}
    bg = {"stiff": (150, 40, 40), "soft": (30, 110, 60)}

    def cap_factory(key):
        env = envs[key]

        def cap(phase, grip_cmd):
            if phase == "settle" and (len(frames[key]) % 3):     # subsample settle for a snappy gif
                pass
            into = env.object_position()[2]
            held = "yes" if env.gripper_holds_object() else "no"
            tilt = np.degrees(env.object_tilt())
            rgb = env.render(camera="planning")
            sub = f"{phase}  |  held={held}  tilt={tilt:.0f}deg  obj_z={into:.3f}"
            frames[key].append(_panel(rgb, title[key], bg[key], sub))
        return cap

    caps = {k: cap_factory(k) for k in envs}

    for row in steps:
        phase = row["phase"]
        for key, env in envs.items():
            if phase.startswith("vjepa"):
                a = np.array([_f(row["dx"]), _f(row["dy"]), _f(row["dz"]),
                              0., 0., 0., _f(row["dgrip"])])
                if np.any(~np.isfinite(a)):
                    continue
                env.apply_action(a)
                caps[key](phase, a[6])
            else:
                if not saw_scripted[key]:
                    saw_scripted[key] = True
                    grasp_error[key] = float(np.linalg.norm(
                        env.object_position()[:2] - env.get_ee_state()[:2]))
                    obj_z0[key] = float(env.object_position()[2])
                if phase == "close":
                    _scripted(env, caps[key], phase, gripper=1.0)
                elif phase == "open":
                    _scripted(env, caps[key], phase, gripper=-1.0)
                elif phase == "lift":
                    _scripted(env, caps[key], phase, dz=0.12)
                elif phase == "lower":
                    cur = env.get_ee_state()[:3]
                    _scripted(env, caps[key], phase, target=[cur[0], cur[1], TABLE_TOP_Z + 0.05])
                elif phase == "settle":
                    _scripted(env, caps[key], phase, settle=settle_n)

    res = {k: _outcome(envs[k], task, arr, grasp_error[k], obj_z0[k]) for k in envs}
    for env in envs.values():
        env.close()

    # align lengths (scripted primitives are identical in count, but guard anyway) and stack
    n = min(len(frames["stiff"]), len(frames["soft"]))
    combined = [np.hstack([frames["stiff"][i], frames["soft"][i]]) for i in range(n)]
    combined += [combined[-1]] * 8
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    imageio.mimsave(out_gif, combined, fps=10, loop=0)

    def tag(r):
        ok, held, tilt, err = r
        return f"{'HIT ' if ok else 'MISS'} held={int(held)} tilt={tilt:.0f} err={err*100:.1f}cm"
    print(f"  {lab} #{trial}: stiff[{tag(res['stiff'])}]  soft[{tag(res['soft'])}] "
          f"-> {os.path.relpath(out_gif, _REPO_ROOT)} ({len(combined)} frames)")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--object", required=True, choices=["cup", "box"])
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--forcelim", type=float, default=0.5)
    p.add_argument("--armkp", type=float, default=None)
    p.add_argument("--gravcomp", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run_dir, rows = _find_group(args.task, args.object)
    if not rows:
        raise SystemExit(f"no data for {args.task}/{args.object}")
    mods = {"forcelim": args.forcelim, "armkp": args.armkp, "gravcomp": args.gravcomp}
    out = args.out or os.path.join(_REPO_ROOT, "results", "demos", "compliance",
                                   f"{args.task}_{args.object}_{args.trial}_stiff_vs_soft.gif")
    render_pair(run_dir, args.task, args.object, args.trial, mods, out)


if __name__ == "__main__":
    main()
