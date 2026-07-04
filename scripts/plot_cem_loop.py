"""Plot a closed-loop CEM episode: latent energy and distance-to-goal per step.

Reads a `cem_loop_*.csv` produced by `scripts/cem_reach_loop.py` and writes an energy-vs-step
and distance-vs-step figure (sub-goals demarcated) to `results/cem_loop/`. This is the graph
for the closed-loop planning experiment; a good episode shows both curves decreasing toward
each sub-goal.

    python scripts/plot_cem_loop.py results/cem_loop/cem_loop_reach.csv
    python scripts/plot_cem_loop.py results/cem_loop/cem_loop_chain.csv --out results/cem_loop
"""
from __future__ import annotations

import argparse
import csv as csvmod
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csvmod.DictReader(fh):
            rows.append({"subgoal": int(r["subgoal"]), "step": int(r["step"]),
                         "energy": float(r["energy"]), "dist": float(r["dist_to_goal_m"]),
                         "reached": int(r["reached"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="a cem_loop_*.csv from cem_reach_loop.py")
    parser.add_argument("--out", default=os.path.join("results", "cem_loop"))
    parser.add_argument("--pos-tol", type=float, default=0.03, help="reached tolerance line (m)")
    args = parser.parse_args()

    rows = load(args.csv)
    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.csv))[0]

    # A single global step axis, with sub-goal boundaries marked.
    xs = list(range(len(rows)))
    energy = [r["energy"] for r in rows]
    dist = [r["dist"] for r in rows]
    boundaries = [i for i in range(1, len(rows)) if rows[i]["subgoal"] != rows[i - 1]["subgoal"]]
    reached_pts = [i for i, r in enumerate(rows) if r["reached"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax1.plot(xs, energy, "-o", color="#1f77b4", label="latent energy |z_obs - z_goal|")
    ax1.set_ylabel("latent energy")
    ax1.set_title(f"Closed-loop CEM planning to goal image(s): {name}")
    ax2.plot(xs, dist, "-o", color="#d62728", label="distance to goal pose (m)")
    ax2.axhline(args.pos_tol, ls="--", color="gray", lw=1, label=f"reach tolerance {args.pos_tol} m")
    ax2.set_ylabel("distance to goal (m)")
    ax2.set_xlabel("planning step (over the episode)")

    for ax in (ax1, ax2):
        for b in boundaries:
            ax.axvline(b - 0.5, color="green", ls=":", lw=1.2)
        for p in reached_pts:
            ax.scatter([p], [energy[p] if ax is ax1 else dist[p]], marker="*",
                       s=180, color="green", zorder=5)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    if boundaries:
        ax1.plot([], [], color="green", ls=":", label="sub-goal boundary")

    fig.tight_layout()
    out_png = os.path.join(args.out, f"{name}.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
