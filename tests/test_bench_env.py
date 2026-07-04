"""Physics regression: the graspable cube env supports a real scripted grasp-lift.

Guards the closed-loop benchmark foundation -- if the cube, gripper contact heuristic, or the
privileged accessors regress, this fails. Skipped automatically if the MuJoCo Menagerie models
are not present.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.envs.franka_build import DEFAULT_MENAGERIE
from src.bench.schema import SUCCESS_DEFAULTS
from src.bench.success import grasp_lift_success

_HAVE_MODELS = os.path.exists(os.path.join(DEFAULT_MENAGERIE, "franka_emika_panda",
                                           "panda_nohand.xml"))


def _goto(env, pos, grip=None, n=4):
    for _ in range(n):
        cur = env.get_ee_state()[:3]
        d = np.zeros(7)
        d[:3] = np.asarray(pos) - cur
        env.apply_action(d)
    if grip is not None:
        for _ in range(3):
            g = np.zeros(7)
            g[6] = grip
            env.apply_action(g)


@pytest.mark.skipif(not _HAVE_MODELS, reason="MuJoCo Menagerie models not present")
def test_scripted_grasp_lift_holds_and_lifts_cube():
    from src.envs.franka_droid_env import FrankaDroidEnv

    env = FrankaDroidEnv(add_object=True, add_zone=True)
    try:
        env.reset()
        # cube starts resting on the table, upright and settled
        cube0 = env.object_position()
        assert abs(cube0[2] - 0.24) < 0.01
        assert env.object_speed() < 0.02
        assert not env.gripper_holds_object()
        assert env.object_released()  # gripper open + not touching cube

        c = cube0
        _goto(env, [c[0], c[1], c[2] + 0.12])          # above cube
        _goto(env, [c[0], c[1], c[2] + 0.005])         # descend
        _goto(env, [c[0], c[1], c[2] + 0.005], grip=1.0)  # close
        z0 = env.object_position()[2]
        assert env.gripper_holds_object()             # pads in contact with cube
        assert not env.object_released()              # held, not released
        _goto(env, [c[0], c[1], c[2] + 0.15])          # lift

        z1 = env.object_position()[2]
        held = env.gripper_holds_object()
        tilt = env.object_tilt()
        assert held
        assert (z1 - z0) > 0.05                        # actually lifted
        assert np.degrees(tilt) < 20.0                 # not toppled

        res = grasp_lift_success(z0, z1, env.get_ee_state()[:2], env.object_position()[:2],
                                 tilt, env.object_speed(), held, SUCCESS_DEFAULTS["grasp_lift"])
        assert res.success, res.metrics
    finally:
        env.close()


@pytest.mark.skipif(not _HAVE_MODELS, reason="MuJoCo Menagerie models not present")
def test_reset_places_cube_at_requested_xy():
    """reset(cube_xy) must honour a randomized start position (benchmark trials randomize)."""
    from src.envs.franka_droid_env import FrankaDroidEnv

    env = FrankaDroidEnv(add_object=True)
    try:
        for xy in [(0.55, -0.05), (0.46, 0.08)]:
            env.reset(cube_xy=xy)
            p = env.object_position()
            assert abs(p[0] - xy[0]) < 0.03 and abs(p[1] - xy[1]) < 0.03  # settled near request
            assert abs(p[2] - 0.24) < 0.01 and env.object_speed() < 0.02  # on table, settled
    finally:
        env.close()


@pytest.mark.skipif(not _HAVE_MODELS, reason="MuJoCo Menagerie models not present")
def test_object_accessors_present_only_with_add_object():
    from src.envs.franka_droid_env import FrankaDroidEnv

    env = FrankaDroidEnv(add_object=False)
    try:
        assert np.all(np.isnan(env.object_pose()))
        assert not env.gripper_holds_object()
    finally:
        env.close()
