"""Unit tests for the hidden success functions (pure; no physics step)."""
from __future__ import annotations

import numpy as np

from src.bench.schema import SUCCESS_DEFAULTS
from src.bench.success import (
    grasp_lift_success,
    place_success,
    reach_success,
    touch_success,
)


def test_reach_success_threshold():
    assert reach_success([0.5, 0.0, 0.3], [0.5, 0.0, 0.32], tau_reach=0.03).success
    r = reach_success([0.5, 0.0, 0.3], [0.5, 0.0, 0.4], tau_reach=0.03)
    assert not r.success and r.failure_type == "too_far"
    assert r.metrics["distance"] > 0.03


def test_touch_success():
    assert touch_success(True, obj_displacement=0.01, move_tol=0.03).success
    assert touch_success(False, 0.0, 0.03).failure_type == "no_contact"
    assert touch_success(True, 0.10, 0.03).failure_type == "knocked_away"


def test_grasp_lift_success_and_failures():
    spec = SUCCESS_DEFAULTS["grasp_lift"]
    good = grasp_lift_success(obj_z0=0.24, obj_z=0.38, ee_xy=[0.5, -0.1], obj_xy=[0.5, -0.1],
                              tilt_rad=np.radians(2.0), speed=0.005, held=True, spec=spec)
    assert good.success and good.failure_type is None
    # never moved / not held -> missed
    missed = grasp_lift_success(0.24, 0.245, [0.5, -0.1], [0.5, -0.1],
                                np.radians(1.0), 0.0, held=False, spec=spec)
    assert not missed.success and missed.failure_type == "missed"
    # lifted + held but toppled -> tipped
    tipped = grasp_lift_success(0.24, 0.34, [0.5, -0.1], [0.5, -0.1],
                                np.radians(60.0), 0.01, held=True, spec=spec)
    assert not tipped.success and tipped.failure_type == "tipped"
    # lifted but object far from gripper -> slipped
    slipped = grasp_lift_success(0.24, 0.34, [0.5, 0.2], [0.5, -0.1],
                                 np.radians(2.0), 0.01, held=False, spec=spec)
    assert not slipped.success and slipped.failure_type == "slipped"


def test_place_success_and_failures():
    spec = SUCCESS_DEFAULTS["place"]
    good = place_success(obj_xy=[0.5, 0.15], zone_xy=[0.5, 0.15], tilt_rad=np.radians(2.0),
                         speed=0.005, released=True, spec=spec)
    assert good.success
    outside = place_success([0.5, 0.30], [0.5, 0.15], np.radians(2.0), 0.005, True, spec)
    assert not outside.success and outside.failure_type == "outside_zone"
    attached = place_success([0.5, 0.15], [0.5, 0.15], np.radians(2.0), 0.005, False, spec)
    assert not attached.success and attached.failure_type == "still_attached"
