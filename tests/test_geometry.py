"""Unit tests for the SO(3) helpers (no GL, no physics)."""
import numpy as np

from src.utils import geometry as geo


def test_identity_roundtrip():
    q = geo.euler_xyz_to_quat(0.0, 0.0, 0.0)
    np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(geo.quat_to_euler_xyz(q), [0, 0, 0], atol=1e-9)


def test_euler_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        roll = rng.uniform(-np.pi, np.pi)
        pitch = rng.uniform(-1.2, 1.2)  # stay away from +/- pi/2 gimbal lock
        yaw = rng.uniform(-np.pi, np.pi)
        q = geo.euler_xyz_to_quat(roll, pitch, yaw)
        back = geo.quat_to_euler_xyz(q)
        np.testing.assert_allclose(back, [roll, pitch, yaw], atol=1e-6)


def test_quat_is_unit():
    q = geo.euler_xyz_to_quat(0.3, -0.7, 1.1)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_quat_mul_identity():
    q = geo.euler_xyz_to_quat(0.2, 0.4, -0.6)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(geo.quat_mul(ident, q), q, atol=1e-12)
    np.testing.assert_allclose(geo.quat_mul(q, ident), q, atol=1e-12)


def test_known_yaw_rotation():
    # 90 deg yaw about Z: rotates world +X to +Y.
    q = geo.euler_xyz_to_quat(0.0, 0.0, np.pi / 2)
    m = geo.quat_to_mat(q)
    np.testing.assert_allclose(m @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)
