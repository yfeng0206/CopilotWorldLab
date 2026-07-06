"""Trace where the pick_place benchmark fails: outcome breakdown, grasp-reach precision on
grasp-failed vs grasp-ok trials, and the held-cube transport under-shoot / stall in the move axis.
Reveals whether the pick-place gap is grasp-budget vs transport (object-salience/frame) limited.

Usage: python scripts/analyze_pick_place_failures.py [logs/closed_loop_runs/<run_id>]
The run dir must contain a pick_place trials.csv + steps.csv (a benchmark run with --tasks pick_place).
"""
import csv, math, statistics as st
import sys
from collections import Counter, defaultdict

RUN = sys.argv[1] if len(sys.argv) > 1 else "logs/closed_loop_runs/20260705_051432"

# --- per-trial outcomes ---
trials = list(csv.DictReader(open(f"{RUN}/trials.csv")))
fails = Counter()
for t in trials:
    fails[t["failure"] or "success"] += 1
print("=== pick_place@200 outcome breakdown (n=%d) ===" % len(trials))
for k, v in fails.most_common():
    print(f"  {k:18s} {v:2d}  ({100*v/len(trials):.0f}%)")

# --- per-step data grouped by trial ---
steps = list(csv.DictReader(open(f"{RUN}/steps.csv")))
by_trial = defaultdict(list)
for s in steps:
    by_trial[int(s["trial"])].append(s)

def f(x):
    try: return float(x)
    except: return None

# grasp reach precision (object-EE xy at the 'close' step) and final placement
grasp_errs, place_errs, drift_x, drift_y = [], [], [], []
grasp_failed_reach, placed_reach = [], []
for tr, rows in by_trial.items():
    outcome = next((t["failure"] for t in trials if int(t["trial"]) == tr), "")
    # grasp reach error = obj_xy - ee_xy at the last vjepa_pnp_grasp step (just before close)
    grasp_rows = [r for r in rows if r["phase"] == "vjepa_pnp_grasp"]
    if grasp_rows:
        r = grasp_rows[-1]
        ox, oy, ex, ey = f(r["obj_x"]), f(r["obj_y"]), f(r["ee_x"]), f(r["ee_y"])
        if None not in (ox, oy, ex, ey):
            ge = math.hypot(ox - ex, oy - ey)
            (grasp_failed_reach if outcome == "grasp_failed" else placed_reach).append(ge)
    # final row: object vs target (zone)
    fin = next((r for r in rows if r["phase"] == "final"), rows[-1])
    ox, oy, tx, ty = f(fin["obj_x"]), f(fin["obj_y"]), f(fin["tgt_x"]), f(fin["tgt_y"])
    if None not in (ox, oy, tx, ty):
        if outcome != "grasp_failed":  # only meaningful if it was carried
            place_errs.append(math.hypot(ox - tx, oy - ty))
            drift_x.append(ox - tx); drift_y.append(oy - ty)

def summ(name, v):
    if v: print(f"  {name}: n={len(v)} mean={100*st.mean(v):.1f}cm median={100*st.median(v):.1f}cm")
    else: print(f"  {name}: (none)")

print("\n=== grasp reach precision (object-EE xy just before scripted close) ===")
summ("grasp-FAILED trials", grasp_failed_reach)
summ("grasp-OK trials     ", placed_reach)

print("\n=== placement error (grasped trials only), and systematic drift ===")
summ("place error", place_errs)
if drift_x:
    print(f"  mean signed drift: dx={100*st.mean(drift_x):+.1f}cm  dy={100*st.mean(drift_y):+.1f}cm")
    print(f"  (object_final - zone; consistent sign => systematic frame/calibration bias)")
    # how many drift same direction
    sx = sum(1 for d in drift_x if d > 0); sy = sum(1 for d in drift_y if d > 0)
    print(f"  dx>0: {sx}/{len(drift_x)}   dy>0: {sy}/{len(drift_y)}")
