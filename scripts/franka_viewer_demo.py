"""Interactive MuJoCo viewer demo: send 7D end-effector actions from Python.

This is the minimal, concrete demonstration of "how MuJoCo takes input". There is
no network, no server, no API: MuJoCo runs inside this Python process, and a command
is delivered by writing numbers into ``data.ctrl`` and calling ``mj_step``. The
interactive viewer window displays the shared ``data`` as we step it.

The signal path for one 7D action ``[dx, dy, dz, droll, dpitch, dyaw, dgripper]``:

    7D EE delta  --(differential IK, src/utils/ik.py)-->  joint targets
        joint targets  -->  data.ctrl[:7]        (arm position servos)
        gripper scalar -->  data.ctrl[grip]      (Robotiq 0..255)
        mj_step(model, data)   x N                (physics advances, servos drive)
        viewer.sync()                             (window redraws the new state)

Run it from the repository root (an interactive desktop session is required for the
GUI window):

    python scripts/franka_viewer_demo.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.franka_build import (  # noqa: E402
    ARM_HOME_QPOS,
    EE_SITE,
    GRIPPER_ACTUATOR,
    build_franka_robotiq,
)
from src.utils import geometry as geo  # noqa: E402
from src.utils import ik as ik_solver  # noqa: E402

ARM_DOF = 7
GRIPPER_CTRL_MAX = 255.0


def main() -> None:
    import mujoco
    import mujoco.viewer

    print("building Franka + Robotiq model ...", flush=True)
    model = build_franka_robotiq()
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)  # scratch for IK forward-kinematics

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)

    # Home the arm and have the position servos hold it.
    data.qpos[:ARM_DOF] = ARM_HOME_QPOS
    data.ctrl[:ARM_DOF] = ARM_HOME_QPOS
    data.ctrl[grip] = 0.0  # gripper open
    mujoco.mj_forward(model, data)

    def ee_pose():
        pos = data.site_xpos[site].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        return pos, quat

    def send_action(viewer, delta7, settle: int = 160, realtime: bool = True) -> None:
        """Deliver one 7D EE-delta 'signal' from Python and step the sim to show it."""
        delta7 = np.asarray(delta7, dtype=np.float64).reshape(7)
        pos, quat = ee_pose()
        target_pos = pos + delta7[:3]
        target_quat = geo.quat_mul(geo.euler_xyz_to_quat(*delta7[3:6]), quat)
        q, pos_err, _ = ik_solver.solve_ik(
            model, ik_data, site, target_pos, geo.quat_normalize(target_quat),
            q_init=data.qpos[:ARM_DOF],
        )
        data.ctrl[:ARM_DOF] = q  # <-- the input: joint targets written to data.ctrl
        g = float(np.clip(data.ctrl[grip] / GRIPPER_CTRL_MAX + delta7[6], 0.0, 1.0))
        data.ctrl[grip] = g * GRIPPER_CTRL_MAX
        for _ in range(settle):
            mujoco.mj_step(model, data)  # <-- physics advances one 2 ms tick
            viewer.sync()                # <-- window redraws shared data
            if realtime:
                time.sleep(model.opt.timestep)
        new_pos, _ = ee_pose()
        print(f"  sent {np.round(delta7, 3)}  ->  EE xyz {np.round(new_pos, 3)} "
              f"(IK residual {pos_err * 1000:.1f} mm)", flush=True)

    # A little choreography that exercises translation, wrist rotation and the gripper.
    sequence = [
        [0.00, 0.00, -0.12, 0, 0, 0, 0.0],   # lower toward the table
        [0.10, 0.00, 0.00, 0, 0, 0, 0.0],    # move +x
        [0.00, -0.10, 0.00, 0, 0, 0, 0.0],   # move -y
        [0.00, 0.00, 0.00, 0, 0, 0.6, 0.0],  # yaw the wrist
        [0.00, 0.00, 0.00, 0, 0, 0, 1.0],    # close gripper
        [0.00, 0.00, 0.15, 0, 0, 0, 0.0],    # lift
        [0.00, 0.00, 0.00, 0, 0, 0, -1.0],   # open gripper
        [-0.10, 0.10, -0.03, 0, 0, -0.6, 0.0],  # return-ish
    ]

    print("launching interactive viewer (close the window to exit) ...", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # settle a moment so the window is up before we start moving
        for _ in range(60):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        print("sending 7D actions from Python:", flush=True)
        for action in sequence:
            if not viewer.is_running():
                break
            send_action(viewer, action)

        print("choreography done; holding pose. Close the window to exit.", flush=True)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("viewer closed.", flush=True)


if __name__ == "__main__":
    main()
