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


def _run_tag(run_dir):
    """The report tag for a run (from run_config report_dir), or '' if unknown."""
    cfgp = os.path.join(run_dir, "run_config.json")
    if not os.path.exists(cfgp):
        return ""
    try:
        with open(cfgp) as fh:
            rep = (json.load(fh).get("logs", {}) or {}).get("report_dir", "")
        # .../results/benchmarks/closed_loop_<tag>/<run_id>
        parts = rep.replace("\\", "/").split("/")
        for i, seg in enumerate(parts):
            if seg.startswith("closed_loop_"):
                return seg[len("closed_loop_"):]
    except (json.JSONDecodeError, OSError):
        pass
    return ""


def _read_trials(run_dir):
    tp = os.path.join(run_dir, "trials.csv")
    if not os.path.exists(tp):
        return []
    with open(tp, newline="") as fh:
        return list(csv.DictReader(fh))


def _gather(run_dir, merge_tag):
    """Return (rows_by_label, config). If merge_tag, aggregate every full800-style run with that
    tag, keeping each (task,object) group from whichever run has the most trials (handles
    pause/resume across run_ids)."""
    if not merge_tag:
        rows = _read_trials(run_dir)
        by = defaultdict(list)
        for r in rows:
            by[r["task"]].append(r)
        cfg = {}
        cfgp = os.path.join(run_dir, "run_config.json")
        if os.path.exists(cfgp):
            with open(cfgp) as fh:
                cfg = json.load(fh)
        return by, cfg, os.path.basename(run_dir)

    best = {}          # label -> (n, rows)
    cfg, run_ids = {}, []
    for d in sorted(glob.glob(os.path.join(_REPO_ROOT, "logs", "closed_loop_runs", "*"))):
        if _run_tag(d) != merge_tag:
            continue
        run_ids.append(os.path.basename(d))
        cfgp = os.path.join(d, "run_config.json")
        if os.path.exists(cfgp) and not cfg:
            with open(cfgp) as fh:
                cfg = json.load(fh)
        by = defaultdict(list)
        for r in _read_trials(d):
            by[r["task"]].append(r)
        for lab, rr in by.items():
            if lab not in best or len(rr) > best[lab][0]:
                best[lab] = (len(rr), rr)
    return {lab: rr for lab, (n, rr) in best.items()}, cfg, ", ".join(run_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="run dir (default: latest under logs/closed_loop_runs)")
    p.add_argument("--tag", default="progress", help="output subdir name under results/benchmarks/")
    p.add_argument("--merge", default=None,
                   help="merge all runs with this report tag (e.g. full800_B), keeping each "
                        "(task,object) group from the run with the most trials -- handles pause/resume")
    args = p.parse_args()

    run_dir = args.run or latest_run()
    if not run_dir or not os.path.isdir(run_dir):
        raise SystemExit("no run dir found")

    agg, cfg, run_id = _gather(run_dir, args.merge)
    thr = defaultdict(lambda: defaultdict(list))
    total = 0
    for lab, rr in agg.items():
        total += len(rr)
        for r in rr:
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
        f"_Snapshot: {datetime.now().isoformat(timespec='seconds')}. "
        f"{'Merged across runs.' if args.merge else 'Live run, still in progress.'}_",
        "",
        f"- config: samples **{cem.get('samples')}**, cem_steps {cem.get('cem_steps')}, "
        f"horizon T={cem.get('rollout_T')}, maxnorm **{cem.get('maxnorm')}**, "
        f"camera **{bun.get('planning_camera')}**, dtype {cfg.get('dtype')}",
        f"- git commit: `{cfg.get('git_commit', '?')[:10]}`",
        f"- trials completed: **{total}** / 500",
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

    print(f"wrote {os.path.relpath(md, _REPO_ROOT)} ({total} trials)")


if __name__ == "__main__":
    main()
