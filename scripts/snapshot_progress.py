"""Snapshot a running fixed-bundle benchmark's live trials.csv into a tracked progress report so
interim results can be committed while the (multi-day) run is still going. Reads the latest (or a
given) run's trials.csv, aggregates per (task, object), and writes a committed markdown + csv under
results/benchmarks/<tag>_progress/.

    python scripts/snapshot_progress.py --tag full800_B
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def latest_run():
    runs = sorted(glob.glob(os.path.join(_REPO_ROOT, "logs", "closed_loop_runs", "*")))
    return runs[-1] if runs else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run dir (default: latest under logs/closed_loop_runs)")
    p.add_argument("--tag", default="progress", help="output subdir name under results/benchmarks/")
    args = p.parse_args()

    run_dir = args.run or latest_run()
    if not run_dir or not os.path.isdir(run_dir):
        raise SystemExit("no run dir found")
    run_id = os.path.basename(run_dir)

    cfg = {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)

    trials_path = os.path.join(run_dir, "trials.csv")
    rows = []
    if os.path.exists(trials_path):
        with open(trials_path, newline="") as fh:
            rows = list(csv.DictReader(fh))

    agg = defaultdict(list)
    thr = defaultdict(lambda: defaultdict(list))
    for r in rows:
        lab = r["task"]
        agg[lab].append(r)
        try:
            for k, v in json.loads(r["success_at_thresholds"]).items():
                thr[lab][k].append(int(v))
        except (KeyError, json.JSONDecodeError):
            pass

    out_dir = os.path.join(_REPO_ROOT, "results", "benchmarks", f"{args.tag}_progress")
    os.makedirs(out_dir, exist_ok=True)

    cem = cfg.get("cem", {})
    bun = cfg.get("bundles", {})
    lines = [
        f"# Benchmark progress -- {args.tag} (run {run_id})",
        "",
        f"_Snapshot: {datetime.now().isoformat(timespec='seconds')}. Live run, still in progress._",
        "",
        f"- config: samples **{cem.get('samples')}**, cem_steps {cem.get('cem_steps')}, "
        f"horizon T={cem.get('rollout_T')}, maxnorm **{cem.get('maxnorm')}**, "
        f"camera **{bun.get('planning_camera')}**, dtype {cfg.get('dtype')}",
        f"- git commit: `{cfg.get('git_commit', '?')[:10]}`",
        f"- trials completed: **{len(rows)}**",
        "",
        "| task / object | n | success@loosest | mean err (cm) | per-threshold success@x |",
        "|---|---|---|---|---|",
    ]
    for lab in sorted(agg):
        rr = agg[lab]
        n = len(rr)
        succ = np.mean([_f(x["success_loose"]) for x in rr]) * 100
        merr = np.nanmean([_f(x["error_m"]) for x in rr]) * 100
        sat = "  ".join(f"@{k}={np.mean(v):.2f}"
                        for k, v in sorted(thr[lab].items(), key=lambda kv: -float(kv[0])))
        lines.append(f"| {lab} | {n} | {succ:.0f}% | {merr:.1f} | {sat} |")

    md = os.path.join(out_dir, "progress.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    csv_out = os.path.join(out_dir, "progress.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["task_object", "n", "success_loose_pct", "mean_err_cm"])
        for lab in sorted(agg):
            rr = agg[lab]
            w.writerow([lab, len(rr),
                        round(np.mean([_f(x["success_loose"]) for x in rr]) * 100, 1),
                        round(np.nanmean([_f(x["error_m"]) for x in rr]) * 100, 2)])

    print(f"wrote {os.path.relpath(md, _REPO_ROOT)} ({len(rows)} trials)")


if __name__ == "__main__":
    main()
