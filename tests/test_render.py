"""Rendering tests. Skipped automatically if no OpenGL context is available
(e.g. a headless CI box). On the target Windows desktop with an RTX 3090 these
should run and pass.
"""
import os

import numpy as np
import pytest

from src.envs.mujoco_scene import MujocoPilotEnv

pytest.importorskip("mujoco")

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "mujoco", "scene.xml")

_GL_HINTS = ("gl", "context", "glfw", "opengl", "display", "egl", "osmesa", "wgl", "framebuffer")


def _make_env():
    """Build an env and force one render; skip if the GL context can't be created."""
    env = MujocoPilotEnv(SCENE, render_width=128, render_height=128)
    try:
        env.render(camera="scene_cam")
    except Exception as exc:  # noqa: BLE001 - we re-raise unless it looks like GL
        env.close()
        if any(h in str(exc).lower() for h in _GL_HINTS):
            pytest.skip(f"No OpenGL context available: {exc}")
        raise
    return env


@pytest.fixture()
def env():
    e = _make_env()
    yield e
    e.close()


def test_render_shape_and_dtype(env):
    img = env.render(camera="scene_cam")
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8


def test_render_is_not_blank(env):
    img = env.render(camera="scene_cam")
    # A real render of a lit scene has colour variation, not a single flat value.
    assert img.std() > 1.0
    assert int(img.max()) > int(img.min())


def test_both_cameras_differ(env):
    scene = env.render(camera="scene_cam")
    wrist = env.render(camera="wrist_cam")
    assert scene.shape == wrist.shape
    assert not np.array_equal(scene, wrist)


def test_custom_size_render(env):
    img = env.render(camera="wrist_cam", width=96, height=72)
    assert img.shape == (72, 96, 3)


def test_capture_goal_image_restores_state(env):
    before = env.get_ee_state()
    img = env.capture_goal_image(pos=[-0.1, -0.1, 0.3], camera="wrist_cam")
    after = env.get_ee_state()
    assert img.shape == (128, 128, 3)
    np.testing.assert_allclose(before, after, atol=1e-9)
