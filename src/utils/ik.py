"""Differential inverse kinematics (damped least squares) for the Franka arm.

Given a target pose for an end-effector site, iteratively solve for the 7 arm joint
angles that place the site there, using the site Jacobian and a damped least-squares
update. Runs on a scratch ``MjData`` (forward kinematics only) so it never disturbs a
live simulation state.

This is the "pose -> joint" layer that lets the arm be driven in end-effector space,
matching V-JEPA 2-AC's 7-D EE-delta action.
"""
from __future__ import annotations

import numpy as np


def solve_ik(
    model,
    ik_data,
    site_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray,
    arm_dof: int = 7,
    iters: int = 100,
    damping: float = 1e-2,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
    step: float = 1.0,
):
    """Return ``(q_arm, pos_err, rot_err)`` placing ``site_id`` at the target pose.

    ``target_quat`` is scalar-first ``[w, x, y, z]``. ``ik_data`` is a scratch
    ``mujoco.MjData`` used purely for forward kinematics.
    """
    import mujoco

    ik_data.qpos[:arm_dof] = np.asarray(q_init, dtype=np.float64).reshape(arm_dof)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    cur_quat = np.zeros(4)
    conj = np.zeros(4)
    err_quat = np.zeros(4)
    err_rot = np.zeros(3)
    lo = model.jnt_range[:arm_dof, 0]
    hi = model.jnt_range[:arm_dof, 1]
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)

    pos_err = rot_err = float("inf")
    for _ in range(iters):
        mujoco.mj_kinematics(model, ik_data)
        mujoco.mj_comPos(model, ik_data)  # required before mj_jacSite

        cur_pos = ik_data.site_xpos[site_id]
        err_pos = target_pos - cur_pos
        mujoco.mju_mat2Quat(cur_quat, ik_data.site_xmat[site_id])
        mujoco.mju_negQuat(conj, cur_quat)
        mujoco.mju_mulQuat(err_quat, target_quat, conj)  # world-frame rotation cur -> target
        mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)

        pos_err = float(np.linalg.norm(err_pos))
        rot_err = float(np.linalg.norm(err_rot))
        if pos_err < pos_tol and rot_err < rot_tol:
            break

        mujoco.mj_jacSite(model, ik_data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, :arm_dof], jacr[:, :arm_dof]])  # 6 x arm_dof
        err = np.concatenate([err_pos, err_rot])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + (damping ** 2) * np.eye(6), err)
        q = np.clip(ik_data.qpos[:arm_dof] + step * dq, lo, hi)
        ik_data.qpos[:arm_dof] = q

    # The in-loop residual is computed before the final update; recompute it for the
    # returned configuration so callers see the true final error (needed to detect a
    # non-converged solve rather than trusting a stale value).
    mujoco.mj_kinematics(model, ik_data)
    mujoco.mj_comPos(model, ik_data)
    mujoco.mju_mat2Quat(cur_quat, ik_data.site_xmat[site_id])
    mujoco.mju_negQuat(conj, cur_quat)
    mujoco.mju_mulQuat(err_quat, target_quat, conj)
    mujoco.mju_quat2Vel(err_rot, err_quat, 1.0)
    pos_err = float(np.linalg.norm(target_pos - ik_data.site_xpos[site_id]))
    rot_err = float(np.linalg.norm(err_rot))

    return ik_data.qpos[:arm_dof].copy(), pos_err, rot_err
