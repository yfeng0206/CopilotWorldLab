"""Tests for the Franka + Robotiq reproduction substrate.

These require the MuJoCo Menagerie models under third_party/ (gitignored). If they
are not present, the whole module is skipped so CI without the vendored models stays
green.
"""
import os

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.envs.franka_build import (  # noqa: E402
    DEFAULT_MENAGERIE,
    EE_SITE,
    build_franka_robotiq,
)

_PANDA = os.path.join(DEFAULT_MENAGERIE, "franka_emika_panda", "panda_nohand.xml")
_ROBOTIQ = os.path.join(DEFAULT_MENAGERIE, "robotiq_2f85", "2f85.xml")
if not (os.path.exists(_PANDA) and os.path.exists(_ROBOTIQ)):
    pytest.skip("MuJoCo Menagerie Franka/Robotiq models not vendored", allow_module_level=True)

_GL_HINTS = ("gl", "context", "glfw", "opengl", "display", "egl", "osmesa", "wgl", "framebuffer")


# ------------------------------------------------------------------ model build
def test_build_compiles_with_expected_structure():
    import mujoco

    model = build_franka_robotiq()
    # 7 arm joints + Robotiq finger joints => at least 13 DoF.
    assert model.nq >= 13
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "exo_cam") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "2f85_fingers_actuator") >= 0


def test_planning_camera_helper_builds_free_camera():
    import mujoco

    from src.envs.franka_build import PLANNING_CAMERA, make_free_camera

    cam = make_free_camera(**PLANNING_CAMERA)
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert cam.azimuth == PLANNING_CAMERA["azimuth"]
    assert cam.distance == PLANNING_CAMERA["distance"]
    np.testing.assert_allclose(cam.lookat, PLANNING_CAMERA["lookat"])


# ----------------------------------------------------------------------- the IK
def test_ik_reaches_reachable_pose():
    import mujoco

    from src.envs.franka_build import ARM_HOME_QPOS
    from src.utils import geometry as geo
    from src.utils import ik as ik_solver

    model = build_franka_robotiq()
    ik_data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)

    target_pos = np.array([0.5, 0.0, 0.35])
    target_quat = geo.euler_xyz_to_quat(np.pi, 0.0, 0.0)  # tool pointing down
    q, pos_err, rot_err = ik_solver.solve_ik(
        model, ik_data, site, target_pos, target_quat, q_init=ARM_HOME_QPOS,
    )
    assert pos_err < 2e-3  # within 2 mm
    assert rot_err < 2e-2
    assert q.shape == (7,)
    lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)


# -------------------------------------------------------------- env kinematics
@pytest.fixture()
def env():
    from src.envs.franka_droid_env import FrankaDroidEnv

    e = FrankaDroidEnv(render_width=64, render_height=64)
    yield e
    e.close()


def test_ee_state_shape(env):
    state = env.get_ee_state()
    assert state.shape == (7,)
    assert state.dtype == np.float32


def test_apply_action_translation(env):
    start = env.get_ee_state()
    end = env.apply_action([0.05, 0.0, 0.0, 0, 0, 0, 0.0])
    # Dynamic servo control: the TCP should reach ~+5 cm within a servo-settle tolerance.
    assert end[0] - start[0] == pytest.approx(0.05, abs=0.015)


def test_reachable_action_reports_ok(env):
    env.apply_action([0.03, 0.0, 0.0, 0, 0, 0, 0.0])
    assert env.last_action_ok is True
    assert env.last_ik_pos_err < env.ik_fail_tol


def test_translation_action_is_bounded(env):
    start = env.get_ee_state()[:3].copy()
    env.apply_action([1.0, 0.0, 0.0, 0, 0, 0, 0.0])  # command 1 m; must be clamped
    moved = float(np.linalg.norm(env.get_ee_state()[:3] - start))
    assert moved <= env.max_translation + 0.02  # within the per-action bound (+ servo slack)


def test_unreachable_action_is_rejected(env):
    ok_flags = []
    for _ in range(12):  # push straight up repeatedly; eventually beyond the workspace
        env.apply_action([0.0, 0.0, env.max_translation, 0, 0, 0, 0.0])
        ok_flags.append(env.last_action_ok)
    assert not all(ok_flags)  # at least one action rejected as unreachable


def test_gripper_command_clamps_and_moves(env):
    start = env.get_ee_state()[6]
    env.apply_action([0, 0, 0, 0, 0, 0, 5.0])   # over-drive close
    assert env._gripper_cmd == pytest.approx(1.0)  # commanded value clamps to [0, 1]
    closed = env.get_ee_state()[6]
    env.apply_action([0, 0, 0, 0, 0, 0, -5.0])  # over-drive open
    assert env._gripper_cmd == pytest.approx(0.0)
    opened = env.get_ee_state()[6]
    assert closed > start + 0.1   # measured opening moved toward closed
    assert opened < closed - 0.1  # and back toward open


def test_capture_goal_image_restores_state(env):
    try:
        env.render(camera="exo_cam")
    except Exception as exc:  # noqa: BLE001
        if any(h in str(exc).lower() for h in _GL_HINTS):
            pytest.skip(f"No OpenGL context available: {exc}")
        raise
    before = env.get_ee_state().copy()
    img = env.capture_goal_image(pos=[0.5, 0.0, 0.3], euler=[np.pi, 0, 0], camera="exo_cam")
    after = env.get_ee_state().copy()
    assert img.shape == (64, 64, 3)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_goal_image_reflects_gripper_state(env):
    try:
        env.render(camera="exo_cam")
    except Exception as exc:  # noqa: BLE001
        if any(h in str(exc).lower() for h in _GL_HINTS):
            pytest.skip(f"No OpenGL context available: {exc}")
        raise
    pose = dict(pos=[0.5, 0.0, 0.3], euler=[np.pi, 0, 0], camera="exo_cam")
    before = env.get_ee_state().copy()
    open_img = env.capture_goal_image(gripper=0.0, **pose).astype(np.int16)
    closed_img = env.capture_goal_image(gripper=1.0, **pose).astype(np.int16)
    after = env.get_ee_state().copy()
    # Settled fingers must change the rendered goal, else open/closed goals alias.
    assert np.abs(open_img - closed_img).max() > 5
    np.testing.assert_allclose(before, after, atol=1e-6)  # live state untouched
