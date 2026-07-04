"""Test whether the V-JEPA 2-AC horizontal action frame is view-relative (offline analysis).

Reads a camera-ablation CSV produced by ``energy_landscape_repro.py --traj`` and, for each
camera, fits the single 2D rotation in the world x-y plane that best maps the ground-truth
action to the energy-minimizing action (over the horizontal transitions). If the model
reasons in a view-relative frame, that fitted angle should track the camera azimuth and a
single rotation should explain the mapping (post-rotation cosine near 1). Vertical z, being
gravity-aligned, should need no such correction.

This is pure offline analysis (no model), used to separate the camera-observability effect
from a fixed, calibratable frame rotation (the paper's App. B.4 W* correction).

    python scripts/analyze_frame_rotation.py                       # latest logs/energy_landscape_*.csv
    python scripts/analyze_frame_rotation.py --csv logs/energy_landscape_XXXX.csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
import glob
import math
import os

import numpy as np

# Nominal free-camera azimuths (deg) used by render_franka_transitions.py, for reference.
CAMERA_AZIMUTH = {
    "az135_el20": -135.0, "az135_el45": -135.0, "az90_el20": -90.0, "az90_el45": -90.0,
    "az45_el20": -45.0, "az45_el45": -45.0, "top_down": -90.0, "exo_named": None,
}
HORIZONTAL_ACTIONS = {"px", "nx", "py", "ny"}


def parse_tag(tag: str):
    """'t_az45_el45_px_p0' -> ('az45_el45', 'px', 0)."""
    body = tag[2:] if tag.startswith("t_") else tag
    parts = body.split("_")
    pose = int(parts[-1][1:]) if parts[-1].startswith("p") else -1
    action = parts[-2]
    camera = "_".join(parts[:-2])
    return camera, action, pose


def fit_plane_rotation(gt_xy: np.ndarray, am_xy: np.ndarray):
    """Least-squares 2D rotation angle mapping gt_xy -> am_xy, and mean post-rotation cosine."""
    s = float(np.sum(gt_xy[:, 0] * am_xy[:, 1] - gt_xy[:, 1] * am_xy[:, 0]))
    c = float(np.sum(gt_xy[:, 0] * am_xy[:, 0] + gt_xy[:, 1] * am_xy[:, 1]))
    theta = math.atan2(s, c)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    rotated = gt_xy @ rot.T
    c008 = []
    for r, a in zip(rotated, am_xy):
        nr, na = np.linalg.norm(r), np.linalg.norm(a)
        if nr > 1e-6 and na > 1e-6:
            c008.append(float(np.dot(r, a) / (nr * na)))
    return math.degrees(theta), (float(np.mean(c008)) if c008 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="ablation CSV (default: latest in logs/)")
    args = parser.parse_args()

    path = args.csv or max(glob.glob(os.path.join("logs", "energy_landscape_*.csv")),
                           key=os.path.getmtime)
    print(f"reading {path}")

    by_cam: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csvmod.DictReader(fh):
            cam, action, _ = parse_tag(row["tag"])
            if action not in HORIZONTAL_ACTIONS:
                continue
            gt = np.array([float(row["gt_x"]), float(row["gt_y"])])
            am = np.array([float(row["min_x"]), float(row["min_y"])])
            by_cam.setdefault(cam, []).append((gt, am))

    print(f"{'camera':<12} {'azimuth':>8} {'fit_rot_deg':>11} {'post_cos':>9} {'n':>3}")
    print("-" * 48)
    rows = []
    for cam, pairs in by_cam.items():
        gt_xy = np.array([g for g, _ in pairs])
        am_xy = np.array([a for _, a in pairs])
        theta, post_cos = fit_plane_rotation(gt_xy, am_xy)
        az = CAMERA_AZIMUTH.get(cam)
        rows.append((cam, az, theta, post_cos, len(pairs)))
    for cam, az, theta, post_cos, n in sorted(rows, key=lambda r: (r[1] is None, r[1] or 0)):
        az_s = "n/a" if az is None else f"{az:.0f}"
        print(f"{cam:<12} {az_s:>8} {theta:>+11.1f} {post_cos:>+9.2f} {n:>3}")

    print("\nInterpretation:")
    print("- If post_cos jumps to ~1 for cameras whose raw alignment was poor, a single")
    print("  in-plane rotation explains the mapping -> horizontal frame is view-relative and")
    print("  calibratable (App. B.4 W*), NOT an unusable camera.")
    print("- If the fitted rotation tracks the camera azimuth, that confirms the view-relative")
    print("  hypothesis. Cameras where post_cos stays low have a genuine observability problem.")


if __name__ == "__main__":
    main()
