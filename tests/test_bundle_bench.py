"""Tests for the fixed-bundle closed-loop benchmark path.

Covers the pure, dependency-light pieces that the runner relies on but that do NOT require loading
the world model: the CLI task/mode guard, the error-aware success/failure classifier, the
threshold single-source-of-truth, and the placement-fair ``object_placed`` env check.

The runner module itself (scripts/run_closed_loop_benchmark.py) rewrites sys.path to the vendored
V-JEPA ``src`` at import time, so it cannot be imported here; the logic under test therefore lives
in src/bench/ (thresholds, success) and src/envs/ where it is importable and reusable.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bench.success import bundle_classify
from src.bench.thresholds import (
    BUNDLE_TASKS,
    LEGACY_TASKS,
    THRESHOLDS,
    validate_task_mode,
)


# --------------------------------------------------------------------- CLI task/mode guard
def test_bundle_only_task_requires_bundles():
    err = validate_task_mode(["grasp"], bundles=None)
    assert err is not None and "require --bundles" in err


def test_legacy_task_rejected_under_bundles():
    err = validate_task_mode(["reach"], bundles="tasks")
    assert err is not None and "supports only" in err


def test_pick_place_valid_in_both_modes():
    assert validate_task_mode(["pick_place"], bundles=None) is None
    assert validate_task_mode(["pick_place"], bundles="tasks") is None


def test_bundle_task_set_valid_under_bundles():
    assert validate_task_mode(BUNDLE_TASKS, bundles="tasks") is None


def test_pick_place_is_the_only_shared_task():
    assert set(LEGACY_TASKS) & set(BUNDLE_TASKS) == {"pick_place"}


# --------------------------------------------------------------------- error-aware classifier
def test_success_requires_error_within_loosest_threshold():
    """All gates pass but the object is outside the loosest sphere -> NOT a success (distance miss)."""
    gates = {"grasped": True, "upright": True, "stable": True, "released": True}
    ok, failure = bundle_classify("pick_place", 0.30, gates, THRESHOLDS["pick_place"])
    assert ok is False and failure == "outside_zone"


def test_success_within_threshold_and_all_gates():
    gates = {"lifted": True, "held": True, "upright": True, "stable": True}
    ok, failure = bundle_classify("grasp", 0.01, gates, THRESHOLDS["grasp"])
    assert ok is True and failure == ""


def test_failed_gate_names_reason_over_distance():
    """A failing gate names the failure even when the object is on target (gate precedence)."""
    gates = {"lifted": True, "held": False, "upright": True, "stable": True}
    ok, failure = bundle_classify("grasp", 0.0, gates, THRESHOLDS["grasp"])
    assert ok is False and failure == "missed"

    gates = {"held": False, "upright": True}
    ok, failure = bundle_classify("reach_with_object", 0.0, gates, THRESHOLDS["reach_with_object"])
    assert ok is False and failure == "dropped"

    gates = {"grasped": False, "upright": True, "stable": True, "released": True}
    ok, failure = bundle_classify("pick_place", 0.0, gates, THRESHOLDS["pick_place"])
    assert ok is False and failure == "grasp_failed"


def test_off_goal_default_reason_for_non_pick_place():
    gates = {"held": True, "upright": True}
    ok, failure = bundle_classify("grasp_and_reach", 0.50, gates, THRESHOLDS["grasp_and_reach"])
    assert ok is False and failure == "off_goal"


def test_classifier_matches_success_at_loosest():
    """bundle_classify's success must equal 'success at the loosest threshold' (consistency with
    the precision-curve success@x, so steps.csv and trials.csv never disagree)."""
    from src.bench.thresholds import success_at
    for task in BUNDLE_TASKS:
        gate_names = [g for g, _ in _all_gate_names(task)]
        for err in (0.0, 0.05, 0.5):
            for flip in [None, *gate_names]:
                gates = {g: True for g in gate_names}
                if flip is not None:
                    gates[flip] = False
                ok, _ = bundle_classify(task, err, gates, THRESHOLDS[task])
                assert ok == success_at(err, gates, max(THRESHOLDS[task]))


def _all_gate_names(task):
    from src.bench.success import _BUNDLE_FAILURE_ORDER
    return _BUNDLE_FAILURE_ORDER[task]


# --------------------------------------------------------------------- threshold single source
def test_generator_sweep_equals_thresholds():
    """The bundle generator must advertise exactly the radii the benchmark scores (no drift)."""
    import scripts.generate_task_bundles as gen
    for task in gen.ALL_TASKS:
        assert list(gen.X_SWEEP[task]) == list(THRESHOLDS[task])


# --------------------------------------------------------------------- placement-fair release
@pytest.mark.parametrize("object_type", ["cup", "box"])
def test_object_placed_true_when_resting_open(object_type):
    """object_placed is True when the object rests at table height with the gripper open, even if a
    finger grazes it (the rim-cup case) -- unlike the strict object_released."""
    from src.envs.franka_droid_env import FrankaDroidEnv
    env = FrankaDroidEnv(render_width=64, render_height=64, add_object=True, add_zone=True,
                         object_type=object_type)
    try:
        env.reset()
        env.place_object(0.5, 0.15, settle=40)      # object resting in the zone
        env.set_ee_pose(pos=[0.5, 0.15, 0.5], euler=[np.pi, 0.0, 0.0], gripper=-1.0)  # arm up, open
        assert env.object_placed() is True
        # lifted off the table -> not placed
        env.set_state(env.data.qpos.copy())
        env.data.qpos[env._cube_qadr + 2] += 0.15
        env._mujoco.mj_forward(env.model, env.data)
        assert env.object_placed() is False
    finally:
        env.close()
