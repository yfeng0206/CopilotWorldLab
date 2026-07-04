"""Generate camera-placement ablation figures and the combined results table.

Reads the ablation CSV (``logs/energy_landscape_*.csv`` from
``energy_landscape_repro.py --traj``) and one rendered frame per camera
(``outputs/transitions/t_<cam>_px_p0.npz``) and writes, into the committed
``results/camera_ablation/`` directory:

- ``camera_grid.png``     -- one rendered view per camera angle, annotated with its
                             azimuth/elevation, mean cosine, fitted frame rotation, and verdict
                             (the "photo of the different camera angles with the result").
- ``camera_ranking.png``  -- per-camera mean-cosine bar chart (best zero-shot view first).
- ``frame_rotation.png``  -- fitted horizontal rotation vs camera azimuth, sized by post-rotation
                             cosine (evidence the horizontal action frame is view-relative).
- ``combined_table.md``   -- per-camera x per-axis cosine + mean + margin + fitted rotation +
                             post-rotation cosine + verdict (the single combined table).

    python scripts/make_ablation_figures.py
    python scripts/make_ablation_figures.py --csv logs/energy_landscape_XXXX.csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
import glob
import math
import os

import numpy as np

# Nominal free-camera (azimuth, elevation) used by render_franka_transitions.py.
CAMERA_INFO = {
    "az135_el20": (-135, -20), "az135_el45": (-135, -45), "az90_el20": (-90, -20),
    "az90_el45": (-90, -45), "az45_el20": (-45, -20), "az45_el45": (-45, -45),
    "top_down": (-90, -85), "exo_named": (None, None),
}
AXES = ["px", "nx", "py", "ny", "pz", "nz"]
HORIZONTAL = {"px", "nx", "py", "ny"}
COS_OK, MARGIN_OK = 0.5, 0.3
RESULTS_DIR = os.path.join("results", "camera_ablation")


def parse_tag(tag: str):
    body = tag[2:] if tag.startswith("t_") else tag
    parts = body.split("_")
    pose = int(parts[-1][1:]) if parts[-1].startswith("p") else -1
    return "_".join(parts[:-2]), parts[-2], pose  # camera, action, pose


def fit_plane_rotation(gt_xy: np.ndarray, am_xy: np.ndarray):
    s = float(np.sum(gt_xy[:, 0] * am_xy[:, 1] - gt_xy[:, 1] * am_xy[:, 0]))
    c = float(np.sum(gt_xy[:, 0] * am_xy[:, 0] + gt_xy[:, 1] * am_xy[:, 1]))
    theta = math.atan2(s, c)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    rotated = gt_xy @ rot.T
    cs = [float(np.dot(r, a) / (np.linalg.norm(r) * np.linalg.norm(a)))
          for r, a in zip(rotated, am_xy)
          if np.linalg.norm(r) > 1e-6 and np.linalg.norm(a) > 1e-6]
    return math.degrees(theta), (float(np.mean(cs)) if cs else 0.0)


def load_rows(csv_path: str):
    per_cam: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csvmod.DictReader(fh):
            cam, action, _ = parse_tag(row["tag"])
            d = per_cam.setdefault(cam, {"cos": {}, "margin": [], "gt_xy": [], "am_xy": []})
            d["cos"].setdefault(action, []).append(float(row["cos_gt"]))
            d["margin"].append(float(row["margin"]))
            if action in HORIZONTAL:
                d["gt_xy"].append([float(row["gt_x"]), float(row["gt_y"])])
                d["am_xy"].append([float(row["min_x"]), float(row["min_y"])])
    return per_cam


def summarize(per_cam: dict):
    out = []
    for cam, d in per_cam.items():
        all_cos = [c for lst in d["cos"].values() for c in lst]
        mean_cos = float(np.mean(all_cos))
        margin = float(np.mean(d["margin"]))
        rot, post = fit_plane_rotation(np.array(d["gt_xy"]), np.array(d["am_xy"]))
        per_axis = {a: (float(np.mean(d["cos"][a])) if a in d["cos"] else float("nan")) for a in AXES}
        verdict = "transfers" if mean_cos >= COS_OK and margin >= MARGIN_OK else "weak"
        out.append({"camera": cam, "mean_cos": mean_cos, "margin": margin, "rot": rot,
                    "post_cos": post, "per_axis": per_axis, "verdict": verdict})
    return sorted(out, key=lambda r: -r["mean_cos"])


def camera_frame(cam: str):
    path = os.path.join("outputs", "transitions", f"t_{cam}_px_p0.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["observations"][0, 0]


def make_camera_grid(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    cols = 4
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(4 * cols, 5.0 * r))
    for ax, row in zip(axes.ravel(), rows):
        cam = row["camera"]
        frame = camera_frame(cam)
        if frame is not None:
            ax.imshow(frame)
        az, el = CAMERA_INFO.get(cam, (None, None))
        az_s = "built-in" if az is None else f"az {az}, el {el}"
        color = "#1a7f37" if row["verdict"] == "transfers" else "#b42318"
        ax.set_title(f"{cam}  ({az_s})\ncos={row['mean_cos']:+.2f}  rot={row['rot']:+.0f} deg"
                     f"  [{row['verdict']}]", color=color, fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Camera-placement ablation: V-JEPA 2-AC zero-shot action alignment per view",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.5)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def make_ranking_bar(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cams = [r["camera"] for r in rows][::-1]
    cos = [r["mean_cos"] for r in rows][::-1]
    colors = ["#1a7f37" if r["verdict"] == "transfers" else "#b42318" for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(cams, cos, color=colors)
    ax.axvline(COS_OK, ls="--", c="gray", lw=1, label=f"transfer threshold ({COS_OK})")
    ax.set_xlabel("mean cosine (energy-min action vs ground-truth action)")
    ax.set_title("Per-camera zero-shot alignment (higher = less interface calibration needed)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def make_rotation_scatter(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        az, _ = CAMERA_INFO.get(row["camera"], (None, None))
        if az is None:
            continue
        size = 60 + 240 * max(0.0, row["post_cos"])
        ax.scatter(az, row["rot"], s=size, alpha=0.8)
        ax.annotate(f"{row['camera']}\npost_cos={row['post_cos']:+.2f}",
                    (az, row["rot"]), textcoords="offset points", xytext=(8, 4), fontsize=8)
    lo, hi = -140, -40
    ax.plot([lo, hi], [-lo - 45, -hi - 45], ls=":", c="gray",
            label="visual guide: rotation ~ -azimuth - 45 (not fitted)")
    ax.set_xlabel("camera azimuth (deg)")
    ax.set_ylabel("fitted horizontal frame rotation (deg)")
    ax.set_title("Fitted W* rotation tracks camera azimuth -> horizontal frame is view-relative")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_combined_table(rows, out_md, baseline_cos):
    """Summary table (best -> worst) with improvement over the built-in-camera baseline."""
    lines = [
        "| camera | az / el | mean cos | improvement vs built-in | margin | fit rot (deg) | post-rot cos | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        az, el = CAMERA_INFO.get(row["camera"], (None, None))
        az_s = "built-in" if az is None else f"{az} / {el}"
        delta = row["mean_cos"] - baseline_cos
        delta_s = "-- (baseline)" if row["camera"] == "exo_named" else f"{delta:+.2f}"
        star = " **(best)**" if row is rows[0] else ""
        lines.append(f"| {row['camera']}{star} | {az_s} | **{row['mean_cos']:+.2f}** | {delta_s} | "
                     f"{row['margin']:.2f} | {row['rot']:+.0f} | {row['post_cos']:+.2f} | "
                     f"{row['verdict']} |")
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def write_per_camera_tables(rows, out_md, baseline_cos):
    """One table per camera angle, ordered worst -> best so the improvement is visible."""
    axis_label = {"px": "+x", "nx": "-x", "py": "+y", "ny": "-y", "pz": "+z", "nz": "-z"}
    blocks = []
    for row in sorted(rows, key=lambda r: r["mean_cos"]):  # worst -> best
        az, el = CAMERA_INFO.get(row["camera"], (None, None))
        az_s = "built-in exo_cam" if az is None else f"azimuth {az} deg, elevation {el} deg"
        delta = row["mean_cos"] - baseline_cos
        tag = " -- BEST" if row is rows[0] else (" -- baseline" if row["camera"] == "exo_named" else "")
        pa = row["per_axis"]
        header = f"#### {row['camera']} ({az_s}){tag}"
        tbl = [
            "| action axis | " + " | ".join(axis_label[a] for a in AXES) + " | mean |",
            "|---|" + "---|" * (len(AXES) + 1),
            "| cosine | " + " | ".join(f"{pa[a]:+.2f}" for a in AXES) + f" | **{row['mean_cos']:+.2f}** |",
        ]
        stats = (f"Energy margin {row['margin']:.2f} | fitted W* rotation {row['rot']:+.0f} deg | "
                 f"post-rotation cos {row['post_cos']:+.2f} | improvement vs built-in "
                 f"{delta:+.2f} | verdict: {row['verdict']}")
        blocks.append(header + "\n\n" + "\n".join(tbl) + "\n\n" + stats + "\n")
    text = "\n".join(blocks)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="ablation CSV (default: latest in logs/)")
    args = parser.parse_args()

    csv_path = args.csv or max(glob.glob(os.path.join("logs", "energy_landscape_*.csv")),
                               key=os.path.getmtime)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"reading {csv_path}")

    rows = summarize(load_rows(csv_path))
    baseline_cos = next((r["mean_cos"] for r in rows if r["camera"] == "exo_named"),
                        min(r["mean_cos"] for r in rows))
    make_camera_grid(rows, os.path.join(RESULTS_DIR, "camera_grid.png"))
    make_ranking_bar(rows, os.path.join(RESULTS_DIR, "camera_ranking.png"))
    make_rotation_scatter(rows, os.path.join(RESULTS_DIR, "frame_rotation.png"))
    table = write_combined_table(rows, os.path.join(RESULTS_DIR, "combined_table.md"), baseline_cos)
    write_per_camera_tables(rows, os.path.join(RESULTS_DIR, "per_camera_tables.md"), baseline_cos)

    best = rows[0]
    print(f"wrote 3 figures + combined_table.md + per_camera_tables.md to {RESULTS_DIR}")
    print(f"baseline (built-in exo_cam) mean cos {baseline_cos:+.2f} -> best {best['camera']} "
          f"{best['mean_cos']:+.2f} (improvement {best['mean_cos'] - baseline_cos:+.2f})\n")
    print(table)


if __name__ == "__main__":
    main()
