"""Figures and a summary table for the transition-scoring benchmark (benchmark_transition_scoring).

Reads a committed benchmark CSV (results/benchmarks/*.csv) and renders, in the I-JEPA_3D_OCT
results style, a two-panel figure and a markdown table:
  1. distribution of per-transition rank_frac (true action vs its random negatives), with the
     mean and the 0.5 chance line;
  2. real goal vs mismatched-goal (null) mean rank_frac -- the image-conditioning control.

    python scripts/plot_transition_benchmark.py \
        --csv results/benchmarks/droid_transition_scoring.csv \
        --title "V-JEPA 2-AC transition scoring -- DROID (n=300)" \
        --out results/benchmarks/droid_transition_scoring
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path: str):
    rows = list(csv.DictReader(open(path)))
    return {
        "rank": np.array([float(r["rank_frac"]) for r in rows]),
        "null": np.array([float(r["null_rank_frac"]) for r in rows]),
        "top1": np.array([int(r["top1"]) for r in rows]),
        "gap_z": np.array([float(r["gap_z"]) for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="results/benchmarks/droid_transition_scoring.csv")
    parser.add_argument("--title", default="V-JEPA 2-AC transition scoring -- DROID")
    parser.add_argument("--out", default="results/benchmarks/droid_transition_scoring")
    args = parser.parse_args()

    d = load(args.csv)
    n = d["rank"].size
    rank_m, null_m, top1_m = d["rank"].mean(), d["null"].mean(), d["top1"].mean()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.hist(d["rank"], bins=20, range=(0, 1), color="#3b7dd8", edgecolor="white", alpha=0.9)
    ax0.axvline(0.5, color="#888", ls="--", lw=1.5, label="chance (0.5)")
    ax0.axvline(rank_m, color="#d1495b", ls="-", lw=2, label=f"mean {rank_m:.3f}")
    ax0.set_xlabel("rank_frac (negatives beaten by the true action)")
    ax0.set_ylabel("transitions")
    ax0.set_title("Per-transition action ranking")
    ax0.legend(frameon=False, fontsize=9)

    bars = ax1.bar(["real goal", "mismatched\ngoal (null)"], [rank_m, null_m],
                   color=["#3b7dd8", "#b0b7c3"], edgecolor="white", width=0.6)
    ax1.axhline(0.5, color="#888", ls="--", lw=1.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("mean rank_frac")
    ax1.set_title("Image-conditioning control")
    for b, v in zip(bars, [rank_m, null_m]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle(f"{args.title}  |  top1={top1_m:.3f}  gap_z={d['gap_z'].mean():+.2f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = f"{args.out}.png"
    fig.savefig(png, dpi=130)
    print("wrote", png)

    table = (
        "| metric | value | meaning |\n"
        "| --- | --- | --- |\n"
        f"| transitions (n) | {n} | real DROID (image_t, state_t) -> (image_t+H) pairs |\n"
        f"| rank_frac | {rank_m:.3f} | mean fraction of {32} random negatives the true xyz action beats (chance 0.5) |\n"
        f"| null rank_frac | {null_m:.3f} | same, but goal from a different episode (image-conditioning control) |\n"
        f"| conditioning gap | {rank_m - null_m:+.3f} | rank_frac - null; the goal-image effect |\n"
        f"| top1_acc | {top1_m:.3f} | fraction of transitions where the true action beats ALL negatives |\n"
        f"| gap_z | {d['gap_z'].mean():+.2f} | mean (neg energy - true energy) / std, effect size |\n"
    )
    md = f"{args.out}_table.md"
    with open(md, "w") as f:
        f.write(table)
    print("wrote", md)
    print(table)


if __name__ == "__main__":
    main()
