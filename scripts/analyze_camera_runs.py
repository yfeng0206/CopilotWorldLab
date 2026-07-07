"""Compare camera runs of the fixed-bundle benchmark on the metrics that matter for grasping.

Reads one or more benchmark run dirs (logs/closed_loop_runs/<id>) and prints a per
(camera, task, object) table: success@x, mean error, EE-to-target 3D/XY/Z error at the end of the
V-JEPA reach, held-after-lift, final object-to-goal error (reach_with_object), first/last latent
energy, and mean planned dx/dy/dz. Object/gripper pixel area is a per-camera property -- read it
from scripts/camera_salience_probe.py (printed separately), not per trial.

    python scripts/analyze_camera_runs.py A=logs/closed_loop_runs/<idA> B=logs/closed_loop_runs/<idB>
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load_steps(run_dir):
    rows = []
    with open(os.path.join(run_dir, "steps.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def _load_trials(run_dir):
    rows = []
    with open(os.path.join(run_dir, "trials.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def analyze(run_dir):
    """Return {(task,object): metrics} for one run."""
    steps = _load_steps(run_dir)
    trials = _load_trials(run_dir)

    # group steps by (task_label, trial)
    by_trial = defaultdict(list)
    for r in steps:
        by_trial[(r["task"], r["trial"])].append(r)

    # aggregate per task-label (task/object)
    agg = defaultdict(lambda: defaultdict(list))
    for (label, ti), rows in by_trial.items():
        vjepa = [r for r in rows if str(r["phase"]).startswith("vjepa")]
        if not vjepa:
            continue
        last = vjepa[-1]
        ee = np.array([_f(last["ee_x"]), _f(last["ee_y"]), _f(last["ee_z"])])
        tgt = np.array([_f(last["tgt_x"]), _f(last["tgt_y"]), _f(last["tgt_z"])])
        agg[label]["ee_tgt_3d"].append(float(np.linalg.norm(ee - tgt)))
        agg[label]["ee_tgt_xy"].append(float(np.linalg.norm(ee[:2] - tgt[:2])))
        agg[label]["ee_tgt_z"].append(float(abs(ee[2] - tgt[2])))
        agg[label]["e_first"].append(_f(vjepa[0]["energy"]))
        agg[label]["e_last"].append(_f(last["energy"]))
        agg[label]["mdx"].append(float(np.nanmean([_f(r["dx"]) for r in vjepa])))
        agg[label]["mdy"].append(float(np.nanmean([_f(r["dy"]) for r in vjepa])))
        agg[label]["mdz"].append(float(np.nanmean([_f(r["dz"]) for r in vjepa])))
        # held after close/lift = held flag on the final recorded row
        agg[label]["held_final"].append(_f(rows[-1]["held"]))

    # success + mean error from trials.csv (authoritative)
    tr = defaultdict(list)
    for r in trials:
        tr[r["task"]].append(r)

    out = {}
    for label in sorted(agg):
        m = agg[label]
        trows = tr.get(label, [])
        errs = [_f(r["error_m"]) for r in trows]
        succ_loose = np.mean([_f(r["success_loose"]) for r in trows]) if trows else float("nan")
        # per-threshold success from the JSON column
        thr = defaultdict(list)
        for r in trows:
            try:
                d = json.loads(r["success_at_thresholds"])
                for k, v in d.items():
                    thr[k].append(int(v))
            except (KeyError, json.JSONDecodeError):
                pass
        out[label] = {
            "n": len(trows),
            "succ_loose": succ_loose,
            "succ_at": {k: float(np.mean(v)) for k, v in sorted(thr.items(), key=lambda kv: -float(kv[0]))},
            "mean_err_cm": float(np.nanmean(errs)) * 100 if errs else float("nan"),
            "ee_tgt_3d_cm": float(np.nanmean(m["ee_tgt_3d"])) * 100,
            "ee_tgt_xy_cm": float(np.nanmean(m["ee_tgt_xy"])) * 100,
            "ee_tgt_z_cm": float(np.nanmean(m["ee_tgt_z"])) * 100,
            "held_pct": float(np.nanmean(m["held_final"])) * 100,
            "e_first": float(np.nanmean(m["e_first"])),
            "e_last": float(np.nanmean(m["e_last"])),
            "mean_dxyz": (float(np.nanmean(m["mdx"])), float(np.nanmean(m["mdy"])),
                         float(np.nanmean(m["mdz"]))),
        }
    return out


def main():
    pairs = []
    for a in sys.argv[1:]:
        if "=" not in a:
            raise SystemExit(f"expected LABEL=run_dir, got {a!r}")
        label, d = a.split("=", 1)
        pairs.append((label, d if os.path.isabs(d) else os.path.join(_REPO_ROOT, d)))
    if not pairs:
        raise SystemExit(__doc__)

    results = {label: analyze(d) for label, d in pairs}
    labels_tasks = sorted({(lab, tk) for lab, res in results.items() for tk in res})

    hdr = (f"{'cam':10s} {'task/obj':22s} {'n':>2s} {'succ%':>6s} {'meanE':>6s} "
           f"{'ee3D':>6s} {'eeXY':>6s} {'eeZ':>6s} {'held%':>6s} {'E0':>6s} {'E1':>6s} "
           f"{'mdx':>7s} {'mdy':>7s} {'mdz':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for lab, tk in labels_tasks:
        r = results[lab].get(tk)
        if not r:
            continue
        dx, dy, dz = r["mean_dxyz"]
        print(f"{lab:10s} {tk:22s} {r['n']:2d} {r['succ_loose'] * 100:6.0f} "
              f"{r['mean_err_cm']:6.1f} {r['ee_tgt_3d_cm']:6.1f} {r['ee_tgt_xy_cm']:6.1f} "
              f"{r['ee_tgt_z_cm']:6.1f} {r['held_pct']:6.0f} {r['e_first']:6.3f} {r['e_last']:6.3f} "
              f"{dx:+7.3f} {dy:+7.3f} {dz:+7.3f}")

    print("\nper-threshold success@x (fraction):")
    for lab, tk in labels_tasks:
        r = results[lab].get(tk)
        if not r:
            continue
        s = "  ".join(f"@{k}={v:.1f}" for k, v in r["succ_at"].items())
        print(f"  {lab:10s} {tk:22s}  {s}")


if __name__ == "__main__":
    main()
