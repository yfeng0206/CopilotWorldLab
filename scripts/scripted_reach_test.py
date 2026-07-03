"""Prescripted reach test in MuJoCo: drive the Franka to a sequence of targets.

A simple, physically-real scripted test on the DROID-style Franka + Robotiq model.
For each prescripted Cartesian target it drives the arm dynamically (the correct
control path: IK -> data.ctrl joint targets -> mj_step, so the position servos and
physics actually move the arm), moves a red marker to the target, and checks whether
the gripper TCP (the Robotiq ``2f85_pinch`` site) reaches it within tolerance. Prints
PASS/FAIL per waypoint and a summary.

This reproduces the paper's "reach" skill with a scripted oracle controller instead of
a world model -- a baseline the world model will later be compared against.

Run from the repository root:

    python scripts/scripted_reach_test.py            # interactive viewer (watch it)
    python scripts/scripted_reach_test.py --headless # no window, saves proof frames
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.franka_build import ARM_HOME_QPOS, EE_SITE, build_franka_robotiq  # noqa: E402
from src.utils import geometry as geo  # noqa: E402
from src.utils import ik as ik_solver  # noqa: E402

ARM_DOF = 7
TOLERANCE = 0.015  # 1.5 cm: "reached" if the TCP is within this of the target
TOOL_DOWN = [np.pi, 0.0, 0.0]  # extrinsic-XYZ Euler: gripper pointing at the table

# Prescripted Cartesian targets (reachable, above the table top at z=0.22).
WAYPOINTS = [
    [0.55, 0.00, 0.35],
    [0.45, 0.15, 0.35],
    [0.45, -0.15, 0.35],
    [0.60, 0.00, 0.30],
    [0.55, 0.00, 0.45],
]


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="no viewer; save frames")
    parser.add_argument("--max-steps", type=int, default=400, help="max sim steps per waypoint")
    args = parser.parse_args()

    model = build_franka_robotiq(add_target=True)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap = int(model.body_mocapid[target_bid])
    target_quat = geo.euler_xyz_to_quat(*TOOL_DOWN)

    data.qpos[:ARM_DOF] = ARM_HOME_QPOS
    data.ctrl[:ARM_DOF] = ARM_HOME_QPOS
    mujoco.mj_forward(model, data)

    def tcp() -> np.ndarray:
        return data.site_xpos[site].copy()

    def drive_to(target_pos, viewer=None) -> float:
        """Dynamically servo the TCP toward target_pos; return final TCP error (m)."""
        data.mocap_pos[target_mocap] = target_pos  # move the red marker
        q, _, _ = ik_solver.solve_ik(
            model, ik_data, site, target_pos, target_quat, q_init=data.qpos[:ARM_DOF],
        )
        data.ctrl[:ARM_DOF] = q  # joint targets; position servos drive the arm there
        err = float("inf")
        for i in range(args.max_steps):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep(model.opt.timestep)
            if i % 20 == 0:
                err = float(np.linalg.norm(tcp() - target_pos))
                if err < TOLERANCE:
                    break
        return float(np.linalg.norm(tcp() - target_pos))

    def run(viewer=None) -> int:
        print(f"prescripted reach test: {len(WAYPOINTS)} targets, tol {TOLERANCE * 100:.1f} cm",
              flush=True)
        passed = 0
        for k, wp in enumerate(WAYPOINTS, 1):
            if viewer is not None and not viewer.is_running():
                break
            err = drive_to(np.asarray(wp, dtype=np.float64), viewer=viewer)
            ok = err < TOLERANCE
            passed += ok
            print(f"  waypoint {k} {np.round(wp, 3)}  TCP err {err * 1000:5.1f} mm  "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        print(f"summary: {passed}/{len(WAYPOINTS)} reached", flush=True)
        return passed

    if args.headless:
        passed = run()
        try:
            os.makedirs("outputs", exist_ok=True)
            import imageio.v2 as imageio

            with mujoco.Renderer(model, height=256, width=256) as r:
                r.update_scene(data, camera="exo_cam")
                imageio.imwrite("outputs/scripted_reach_final.png", r.render())
            print("saved outputs/scripted_reach_final.png", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"(render skipped: {exc})", flush=True)
        sys.exit(0 if passed == len(WAYPOINTS) else 1)
    else:
        import mujoco.viewer

        print("launching interactive viewer (close the window to exit) ...", flush=True)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for _ in range(60):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
            run(viewer)
            print("done; holding pose. Close the window to exit.", flush=True)
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
