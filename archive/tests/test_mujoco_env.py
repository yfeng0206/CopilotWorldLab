"""Environment tests that do not need an OpenGL context (model + kinematics)."""
import os

import numpy as np
import pytest

from src.envs.mujoco_scene import MujocoPilotEnv, STATE_DIM

pytest.importorskip("mujoco")

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "mujoco", "scene.xml")


@pytest.fixture()
def env():
    e = MujocoPilotEnv(SCENE, render_width=64, render_height=64)
    yield e
    e.close()


def test_model_loads_and_has_mocap(env):
    assert env.model is not None
    assert env._ee_mocap >= 0


def test_ee_state_shape(env):
    state = env.get_ee_state()
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32


def test_apply_action_translates(env):
    start = env.get_ee_state()
    env.apply_action([0.03, 0.0, 0.0, 0, 0, 0, 0], step_physics=False)
    end = env.get_ee_state()
    assert end[0] - start[0] == pytest.approx(0.03, abs=1e-6)


def test_gripper_clamps(env):
    env.apply_action([0, 0, 0, 0, 0, 0, 5.0], step_physics=False)
    assert env.get_ee_state()[6] == pytest.approx(1.0)
    env.apply_action([0, 0, 0, 0, 0, 0, -5.0], step_physics=False)
    assert env.get_ee_state()[6] == pytest.approx(0.0)


def test_set_ee_pose_roundtrip(env):
    env.set_ee_pose(pos=[0.1, -0.05, 0.4], euler=[0.0, 0.0, np.pi / 4], gripper=0.5)
    state = env.get_ee_state()
    np.testing.assert_allclose(state[:3], [0.1, -0.05, 0.4], atol=1e-6)
    assert state[5] == pytest.approx(np.pi / 4, abs=1e-5)
    assert state[6] == pytest.approx(0.5)
