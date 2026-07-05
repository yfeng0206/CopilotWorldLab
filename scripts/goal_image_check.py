"""Standalone goal-image verification (no V-JEPA/GPU): render the goal images each protocol/task
feeds to CEM, with red=object / blue=EE-goal / green=zone markers, into one contact sheet. This is
an auditability artifact proving the held-object place goal shows the cube carried over the zone
(not left behind). Uses the exact PLANNING_CAMERA projection from the benchmark runner.
"""
import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.franka_droid_env import FrankaDroidEnv
from src.envs.franka_build import PLANNING_CAMERA, TABLE_TOP_Z, CUBE_HALF
from src.bench.schema import SUCCESS_DEFAULTS

EE_DOWN = [np.pi, 0.0, 0.0]
CROP = 256


def _cam_basis(cam=PLANNING_CAMERA):
    az, el = np.radians(cam["azimuth"]), np.radians(cam["elevation"])
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    cam_pos = np.asarray(cam["lookat"], float) - cam["distance"] * forward
    right = np.cross(forward, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return cam_pos, forward, right, up


def project(points, fovy, H=CROP, W=CROP, cam=PLANNING_CAMERA):
    cam_pos, forward, right, up = _cam_basis(cam)
    f = 0.5 * H / np.tan(0.5 * np.radians(fovy))
    out = []
    for p in np.atleast_2d(np.asarray(points, float)):
        d = p - cam_pos
        z = d @ forward
        out.append((np.nan, np.nan) if z <= 1e-6
                   else (W / 2 + f * (d @ right) / z, H / 2 - f * (d @ up) / z))
    return np.asarray(out)


def grasp_cube(env):
    """Reliable dynamic grasp (mirrors runner _scripted_grasp) so held-object goals have a real grip."""
    c = env.object_position()
    for tgt in ([c[0], c[1], c[2] + 0.12], None):
        if tgt is None:
            c = env.object_position(); tgt = [c[0], c[1], c[2] + 0.005]
        for _ in range(4):
            d = np.zeros(7); d[:3] = np.asarray(tgt) - env.get_ee_state()[:3]; env.apply_action(d)
    for _ in range(3):
        g = np.zeros(7); g[6] = 1.0; env.apply_action(g)
    for _ in range(4):
        d = np.zeros(7); d[2] = 0.12 / 4; env.apply_action(d)


def main():
    env = FrankaDroidEnv(render_width=CROP, render_height=CROP, add_object=True, add_zone=True)
    fovy = float(env.model.vis.global_.fovy)
    zr = SUCCESS_DEFAULTS["place"]["zone_radius"]

    panels = []  # (title, img, ee_goal_xyz, obj_xyz_in_goal)

    # --- grasp goals: cube UNCHANGED on the table, only the arm moves ---
    env.reset(cube_xy=(0.50, -0.10))
    c = env.object_position()
    pregrasp = np.array([c[0], c[1], c[2] + 0.10]); grasp = np.array([c[0], c[1], c[2] + 0.005])
    panels.append(("grasp multistage: PREGRASP goal (arm above cube)",
                   env.capture_goal_image(pos=pregrasp, euler=EE_DOWN, gripper=0.0, camera="planning"),
                   pregrasp, c.copy()))
    panels.append(("grasp: GRASP goal (arm around cube)",
                   env.capture_goal_image(pos=grasp, euler=EE_DOWN, gripper=0.0, camera="planning"),
                   grasp, c.copy()))

    # --- place goals: cube HELD, carried over the zone ---
    env.reset(cube_xy=(0.50, -0.10))
    grasp_cube(env)
    zone = env.zone_center()
    hover = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.12])
    low = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.04])
    held = env.gripper_holds_object()
    panels.append((f"place single_goal / vicinity: HELD goal over zone (held={held})",
                   env.capture_goal_image(pos=hover, euler=EE_DOWN, gripper=1.0, camera="planning",
                                          held_object=True), hover, hover))
    panels.append(("place multistage: FINAL goal (held cube lowered onto zone)",
                   env.capture_goal_image(pos=low, euler=EE_DOWN, gripper=1.0, camera="planning",
                                          held_object=True), low, low))

    zone3 = np.array([zone[0], zone[1], TABLE_TOP_Z + 0.001])
    zc = project(zone3, fovy)[0]
    zr_px = float(np.linalg.norm(project([zone3, zone3 + [zr, 0, 0]], fovy)[1] - zc))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (title, img, ee, obj) in zip(axes.ravel(), panels):
        ax.imshow(img); ax.set_xlim(0, CROP); ax.set_ylim(CROP, 0); ax.axis("off")
        ax.set_title(title, fontsize=9)
        ax.add_patch(plt.Circle(tuple(zc), max(zr_px, 3.0), fill=False, color="lime", lw=2.0))
        ep = project(ee, fovy)[0]; op = project(obj, fovy)[0]
        ax.plot(ep[0], ep[1], "o", color="blue", ms=10, mec="white", mew=1.3)
        ax.plot(op[0], op[1], "o", color="red", ms=8, mec="white", mew=1.2)
    fig.suptitle("Goal-image check: red=object-in-goal, blue=EE-goal, green=zone "
                 "(PLANNING_CAMERA projection)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "results/benchmarks/closed_loop_smoke/goal_image_check.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out, "| zone", np.round(zone, 3), "| held", held)


if __name__ == "__main__":
    main()
