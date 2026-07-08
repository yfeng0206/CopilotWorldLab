"""Estimate a mechanics fix from the logs, without re-running V-JEPA.

Because the arm is position-controlled (IK to an EE target), replaying a trial's *logged* EE actions
reproduces the same arm trajectory regardless of the (light) manipuland. So we can restore each
scenario, replay the logged actions under a MODIFIED object (friction / mass / contact softness),
re-run the same scripted gripper tail, and re-score the exact benchmark gates. Comparing the
replayed success at baseline (no change) vs a candidate fix estimates how much the fix would move the
benchmark -- from existing logs, no GPU.

    python scripts/rescore_from_log.py --task grasp --object box            # baseline reproduction
    python scripts/rescore_from_log.py --task grasp --object box --mass 4 --soften   # a candidate fix
"""
from __future__ import annotations

import argparse
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
_THR = {"grasp": 0.06, "reach_with_object": 0.10, "grasp_and_reach": 0.10,
        "pick_place": 0.10, "place_with_object": 0.10}   # loosest sphere per task


def _run_tag(d):
    p = os.path.join(d, "run_config.json")
    if not os.path.exists(p):
        return ""
    rep = (json.load(open(p)).get("logs", {}) or {}).get("report_dir", "")
    for s in rep.replace("\\", "/").split("/"):
        if s.startswith("closed_loop_"):
            return s[len("closed_loop_"):]
    return ""


def _find_group(task, obj, tag="full800_B"):
    lab = f"{task}/{obj}"
    best, rows = None, []
    for d in sorted(glob.glob(os.path.join(_REPO_ROOT, "logs", "closed_loop_runs", "*"))):
        if _run_tag(d) != tag:
            continue
        tc = os.path.join(d, "trials.csv")
        if os.path.exists(tc):
            rr = [r for r in csv.DictReader(open(tc)) if r["task"] == lab]
            if len(rr) > len(rows):
                best, rows = d, rr
    return best, rows


_ARM_DOF = 7   # first 7 actuators are the Franka arm joints; the rest are the gripper


def _apply_mods(env, mods):
    """Modify object physics and/or arm compliance on the built model.

    Object knobs act on the manipuland (friction / contact softness / mass). The arm-compliance
    knobs act on the 7 arm position servos and are the "make it behave like a real robot" fix:
      - forcelim: scale the arm actuator force cap. The real Franka limits (+/-87 / +/-12 Nm) are
        plenty to shove a ~50 g object through a soft contact; a real robot avoids that via its
        controller, not weak motors. Scaling the cap down makes the servo STALL at contact instead
        of bulldozing, while free-space reaching (forces far below the cap) is unchanged.
      - armkp: scale the position-servo stiffness (gain + matching bias). Softer = more compliant
        on contact, but sags more under gravity, so keep it moderate.
    """
    m = env.model
    cube_bid = env._cube_bid
    gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == cube_bid]
    if mods.get("friction"):
        for g in gids:
            m.geom_friction[g, 0] *= mods["friction"]
    if mods.get("soften"):
        # softer, more damped normal contact: longer solref timeconst, wider solimp band
        for g in gids:
            m.geom_solref[g] = [0.02, 1.5]
            m.geom_solimp[g] = [0.9, 0.98, 0.001, 0.5, 2.0]
    if mods.get("mass"):
        if cube_bid >= 0 and m.body_mass[cube_bid] > 0:
            f = mods["mass"]
            m.body_mass[cube_bid] *= f
            m.body_inertia[cube_bid] *= f
    if mods.get("gravcomp"):
        mj = env._mujoco
        for b in range(m.nbody):
            nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith("link") or nm == "attachment" or nm.startswith("2f85"):
                m.body_gravcomp[b] = 1.0   # float the arm/gripper so the force budget is free
    if mods.get("forcelim"):
        s = mods["forcelim"]
        for i in range(_ARM_DOF):
            m.actuator_forcerange[i] *= s
            m.actuator_forcelimited[i] = 1
    if mods.get("armkp"):
        s = mods["armkp"]
        for i in range(_ARM_DOF):
            m.actuator_gainprm[i, 0] *= s
            m.actuator_biasprm[i, 1] *= s   # keep bias = -gain so it stays a proper position servo
    env._mujoco.mj_forward(m, env.data)


def score_trial(run_dir, task, obj, trial, mods, cam_override):
    from src.envs.franka_build import TABLE_TOP_Z, OBJECT_SPECS
    from src.envs.franka_droid_env import FrankaDroidEnv

    lab = f"{task}/{obj}"
    bundle_id = next(r["bundle_id"] for r in csv.DictReader(open(os.path.join(run_dir, "trials.csv")))
                     if r["task"] == lab and int(r["trial"]) == trial)
    steps = [r for r in csv.DictReader(open(os.path.join(run_dir, "steps.csv")))
             if r["task"] == lab and int(r["trial"]) == trial]
    bdir = os.path.join(_REPO_ROOT, "tasks", task, obj, bundle_id)
    arr = dict(np.load(os.path.join(bdir, "arrays.npz")))
    meta = json.load(open(os.path.join(bdir, "meta.json")))
    start_grasped = bool(meta.get("start_grasped", False))

    env = FrankaDroidEnv(render_width=64, render_height=64, add_object=True, add_zone=True,
                         object_type=obj, add_distractors=True, planning_camera=cam_override)
    _apply_mods(env, mods)

    env.reset()
    zone = arr.get("zone")
    if zone is not None and np.all(np.isfinite(zone)):
        env.set_zone_xy(float(zone[0]), float(zone[1]))
    env.set_state(arr["qpos0"], gripper=(1.0 if start_grasped else 0.0),
                  settle=(8 if start_grasped else 0))

    # optional descent z-floor guard: the executed EE must not drop below the object's grasp height,
    # so the gripper stops at the object instead of driving it into the table.
    zfloor = None
    if mods.get("zfloor") is not None:
        zfloor = TABLE_TOP_Z + OBJECT_SPECS[obj]["rest_half_z"] + mods["zfloor"]

    settle_n = _SETTLE.get(task, 20)
    grasp_error = None
    obj_z0 = None
    saw_scripted = False

    def noop(*a):
        pass

    for row in steps:
        phase = row["phase"]
        if phase.startswith("vjepa"):
            a = np.array([_f(row["dx"]), _f(row["dy"]), _f(row["dz"]), 0., 0., 0., _f(row["dgrip"])])
            if np.any(~np.isfinite(a)):
                continue
            if zfloor is not None:
                ez = float(env.get_ee_state()[2])
                if ez + a[2] < zfloor:
                    a[2] = max(0.0, zfloor - ez)   # clamp the descent, never force it up
            env.apply_action(a)
        else:
            if not saw_scripted:
                saw_scripted = True
                # capture the grasp-moment quantities (matches run_bundle_trial)
                grasp_error = float(np.linalg.norm(env.object_position()[:2] - env.get_ee_state()[:2]))
                obj_z0 = float(env.object_position()[2])
            if phase == "close":
                _scripted(env, noop, phase, gripper=1.0)
            elif phase == "open":
                _scripted(env, noop, phase, gripper=-1.0)
            elif phase == "lift":
                _scripted(env, noop, phase, dz=0.12)
            elif phase == "lower":
                cur = env.get_ee_state()[:3]
                _scripted(env, noop, phase, target=[cur[0], cur[1], TABLE_TOP_Z + 0.05])
            elif phase == "settle":
                _scripted(env, noop, phase, settle=settle_n)

    obj = env.object_position()
    tilt = float(np.degrees(env.object_tilt()))
    held = bool(env.gripper_holds_object())
    speed = float(env.object_speed())
    if task == "grasp":
        error = grasp_error if grasp_error is not None else 1.0
        lifted = (obj[2] - (obj_z0 if obj_z0 is not None else obj[2])) > 0.04
        gates = {"lifted": lifted, "held": held, "upright": tilt < 30.0, "stable": speed < 0.05}
        if mods.get("grasp_no_upright"):
            # paper grasp success = gripped + lifted; a gripped-but-tilted object still counts
            gates = {"lifted": lifted, "held": held}
    elif task in ("reach_with_object", "grasp_and_reach"):
        goal_obj = np.asarray(arr["goal_object"], dtype=float)
        error = float(np.linalg.norm(obj - goal_obj))
        gates = {"held": held, "upright": tilt < 30.0}
    else:  # pick_place / place_with_object
        error = float(np.linalg.norm(obj[:2] - env.zone_center()))
        gates = {"upright": tilt < 25.0, "stable": speed < 0.05,
                 "released": bool(env.object_placed())}
        if task == "pick_place":
            gates["grasped"] = True   # not recoverable from log; assume as-scored
    env.close()
    ok = error < _THR[task] and all(gates.values())
    return ok, tilt, held, error


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--object", required=True, choices=["cup", "box"])
    p.add_argument("--limit", type=int, default=None, help="only the first N trials (quick check)")
    p.add_argument("--friction", type=float, default=None, help="scale object geom friction")
    p.add_argument("--mass", type=float, default=None, help="scale object mass + inertia")
    p.add_argument("--soften", action="store_true", help="softer, damped object contact")
    p.add_argument("--zfloor", type=float, default=None,
                   help="descent guard: clamp executed EE z to >= object_rest_z + this margin (m), "
                        "so the gripper cannot drive the object into the table (e.g. 0.0 or 0.005)")
    p.add_argument("--forcelim", type=float, default=None,
                   help="scale the arm actuator force cap (compliance): <1 makes the servo stall at "
                        "contact instead of bulldozing (e.g. 0.5, 0.3, 0.2)")
    p.add_argument("--armkp", type=float, default=None,
                   help="scale the arm position-servo stiffness (gain+bias): <1 is softer/compliant")
    p.add_argument("--gravcomp", action="store_true",
                   help="gravity-compensate the arm/gripper bodies (passive float) so the force cap is "
                        "free to be gentle on contact; pair with --forcelim/--armkp for clean compliance")
    p.add_argument("--grasp-no-upright", action="store_true",
                   help="score grasp as gripped+lifted only (drop the upright/stable gates), matching "
                        "the paper's 'grip the object' definition")
    args = p.parse_args()

    run_dir, rows = _find_group(args.task, args.object)
    if not rows:
        raise SystemExit(f"no data for {args.task}/{args.object}")
    cfg = json.load(open(os.path.join(run_dir, "run_config.json")))
    cam_override = _PRESETS.get((cfg.get("bundles", {}) or {}).get("planning_camera"))

    trials = sorted(int(r["trial"]) for r in rows)
    if args.limit:
        trials = trials[:args.limit]
    logged = {int(r["trial"]): int(r["success_loose"]) for r in rows}
    mods = {"friction": args.friction, "mass": args.mass, "soften": args.soften,
            "zfloor": args.zfloor, "forcelim": args.forcelim, "armkp": args.armkp,
            "gravcomp": args.gravcomp, "grasp_no_upright": args.grasp_no_upright}
    modstr = ", ".join(f"{k}={v}" for k, v in mods.items() if v is not None and v is not False) \
        or "BASELINE (no change)"

    n_log = sum(logged[t] for t in trials)
    n_rep = 0
    flips = []
    for t in trials:
        ok, tilt, held, err = score_trial(run_dir, args.task, args.object, t, mods, cam_override)
        n_rep += int(ok)
        if int(ok) != logged[t]:
            flips.append((t, logged[t], int(ok), tilt, held, err))

    print(f"\n{args.task}/{args.object}  ({len(trials)} trials)  mods: {modstr}")
    print(f"  logged success:            {n_log}/{len(trials)} ({100*n_log/len(trials):.0f}%)")
    print(f"  replayed success:          {n_rep}/{len(trials)} ({100*n_rep/len(trials):.0f}%)")
    print(f"  flips vs logged:           {len(flips)}")
    for t, lo, rp, tilt, held, err in flips[:12]:
        arrow = "MISS->HIT" if rp > lo else "HIT->MISS"
        print(f"    trial {t:>2} {arrow}  final_tilt={tilt:5.1f}  held={int(held)}  err={err*100:.1f}cm")


if __name__ == "__main__":
    main()
