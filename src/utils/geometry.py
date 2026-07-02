"""Small, dependency-light SO(3) helpers.

Conventions
-----------
- Quaternions are scalar-first ``[w, x, y, z]``, matching MuJoCo's ``mocap_quat``.
- Euler angles are extrinsic XYZ (rotate about fixed world X, then Y, then Z),
  which is the representation V-JEPA 2-AC uses for the end-effector orientation
  ("extrinsic Euler angles", arXiv:2506.09985). The composed rotation matrix is
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

These helpers are pure NumPy so they can be unit-tested without a GL context or
a physics step.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions (a followed by b)."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic XYZ Euler angles (radians) to a scalar-first unit quaternion."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    qx = np.array([cr, sr, 0.0, 0.0])
    qy = np.array([cp, 0.0, sp, 0.0])
    qz = np.array([cy, 0.0, 0.0, sy])
    # extrinsic XYZ => R = Rz @ Ry @ Rx => q = qz * qy * qx
    return quat_normalize(quat_mul(qz, quat_mul(qy, qx)))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Scalar-first unit quaternion to extrinsic XYZ Euler angles (radians).

    Inverse of :func:`euler_xyz_to_quat` away from the pitch = +/- pi/2
    singularity. Returns ``[roll, pitch, yaw]``.
    """
    m = quat_to_mat(q)
    # For R = Rz @ Ry @ Rx, m[2,0] = -sin(pitch).
    sp = -m[2, 0]
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: fix roll = 0
        roll = 0.0
        yaw = np.arctan2(-m[0, 1], m[1, 1])
    return np.array([roll, pitch, yaw])
