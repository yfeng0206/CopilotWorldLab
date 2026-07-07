"""Hidden success functions for the closed-loop task-success benchmark.

The model only ever sees observations (image + EE state + goal image); the *evaluator* uses
these functions on privileged simulator truth (object pose, contacts, velocity, tilt) to decide
success. They are pure -- given the privileged scalars/arrays they return a verdict -- so they
are unit-testable without a physics step; the benchmark runner gathers the values from
``FrankaDroidEnv`` (object_pose/speed/tilt, zone_center, gripper_holds_object) and calls them.

Thresholds default to ``src.bench.schema.SUCCESS_DEFAULTS`` and are calibrated against scripted
expert rollouts (a good grasp-lift shows tilt ~2 deg, speed ~0.005 m/s, obj_dz ~0.14 m). See
docs/experiments/closed_loop_success_plan.md Section 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SuccessResult:
    success: bool
    failure_type: str | None  # None on success; else a category (Section 3 of the plan)
    metrics: dict


def reach_success(ee_pos, target_pos, tau_reach: float) -> SuccessResult:
    dist = float(np.linalg.norm(np.asarray(ee_pos, dtype=float)[:3]
                                - np.asarray(target_pos, dtype=float)[:3]))
    ok = dist < tau_reach
    return SuccessResult(ok, None if ok else "too_far", {"distance": dist})


def touch_success(in_contact: bool, obj_displacement: float, move_tol: float) -> SuccessResult:
    ok = bool(in_contact) and obj_displacement < move_tol
    ft = None if ok else ("knocked_away" if obj_displacement >= move_tol else "no_contact")
    return SuccessResult(ok, ft, {"displacement": float(obj_displacement),
                                  "contact": bool(in_contact)})


def grasp_lift_success(obj_z0: float, obj_z: float, ee_xy, obj_xy, tilt_rad: float,
                       speed: float, held: bool, spec: dict) -> SuccessResult:
    """Object lifted, still held near the gripper, upright, and settled."""
    lift_dz = spec.get("lift_dz", 0.04)
    grasp_radius = spec.get("grasp_radius", 0.05)
    tilt_max = np.radians(spec.get("tilt_max_deg", 30.0))
    v_settle = spec.get("v_settle", 0.05)

    dz = float(obj_z - obj_z0)
    lateral = float(np.linalg.norm(np.asarray(ee_xy, dtype=float) - np.asarray(obj_xy, dtype=float)))
    lifted = dz > lift_dz
    near = lateral < grasp_radius
    upright = tilt_rad < tilt_max
    stable = speed < v_settle
    ok = lifted and near and upright and stable and bool(held)

    ft = None
    if not ok:
        if not held and not lifted:
            ft = "missed" if dz < 0.5 * lift_dz else "dropped"
        elif not lifted:
            ft = "pushed"
        elif not near or not held:
            ft = "slipped"
        elif not upright:
            ft = "tipped"
        else:
            ft = "unstable"
    return SuccessResult(ok, ft, {"obj_dz": dz, "lateral": lateral,
                                  "tilt_deg": float(np.degrees(tilt_rad)),
                                  "speed": float(speed), "held": bool(held)})


def place_success(obj_xy, zone_xy, tilt_rad: float, speed: float, released: bool,
                  spec: dict) -> SuccessResult:
    """Object resting inside the target zone, upright, settled, and released."""
    zone_radius = spec.get("zone_radius", 0.06)
    tilt_max = np.radians(spec.get("tilt_max_deg", 25.0))
    v_settle = spec.get("v_settle", 0.05)

    dist = float(np.linalg.norm(np.asarray(obj_xy, dtype=float) - np.asarray(zone_xy, dtype=float)))
    in_zone = dist < zone_radius
    upright = tilt_rad < tilt_max
    stable = speed < v_settle
    ok = in_zone and upright and stable and bool(released)

    ft = None
    if not ok:
        if not in_zone:
            ft = "outside_zone"
        elif not released:
            ft = "still_attached"
        elif not upright:
            ft = "tipped"
        else:
            ft = "unstable"
    return SuccessResult(ok, ft, {"zone_dist": dist, "tilt_deg": float(np.degrees(tilt_rad)),
                                  "speed": float(speed), "released": bool(released)})


# --- fixed-bundle task classifier (grasp / reach_with_object / grasp_and_reach / pick_place) -------
# Ordered (gate -> failure reason) per task: the FIRST failed gate names the failure. If every gate
# holds but the object is outside the loosest precision sphere, it is a pure distance miss.
_BUNDLE_FAILURE_ORDER = {
    "grasp": [("held", "missed"), ("lifted", "not_lifted"),
              ("upright", "tipped"), ("stable", "unstable")],
    "reach_with_object": [("held", "dropped"), ("upright", "tipped")],
    "grasp_and_reach": [("held", "dropped"), ("upright", "tipped")],
    "pick_place": [("grasped", "grasp_failed"), ("released", "not_released"),
                   ("upright", "tipped"), ("stable", "unstable")],
    "place_with_object": [("released", "not_released"), ("upright", "tipped"),
                          ("stable", "unstable")],
}
_BUNDLE_OFF_GOAL = {"pick_place": "outside_zone", "place_with_object": "outside_zone"}  # default: "off_goal"


def bundle_classify(task: str, error: float, gates: dict, thresholds) -> tuple[bool, str]:
    """Final verdict for a fixed-bundle trial: success requires the object within the LOOSEST
    precision sphere (``max(thresholds)``) AND every physical gate. This keeps the per-step
    ``success`` flag, the ``failure`` label, and the precision-curve ``success@x`` mutually
    consistent (all error-aware) rather than gates-only. Returns ``(success, failure)`` with
    ``failure=""`` on success.
    """
    ok = bool(error < max(thresholds) and all(gates.values()))
    if ok:
        return True, ""
    for gate, reason in _BUNDLE_FAILURE_ORDER.get(task, []):
        if not gates.get(gate, True):
            return False, reason
    return False, _BUNDLE_OFF_GOAL.get(task, "off_goal")
