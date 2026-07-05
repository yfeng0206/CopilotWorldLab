"""Precision thresholds, physical gates, and the success rule for the closed-loop task-success
benchmark. Kept dependency-light (pure Python) so the success logic is importable and unit-testable
without loading the world model or the vendored V-JEPA namespace.

A trial records one continuous error; success at precision threshold ``t`` requires ``error < t``
AND every physical gate in ``GATE_SPEC[task]`` to hold. The gates guard against precision-only false
positives (e.g. a pick_place trial whose cube lands near the zone but was never actually grasped).
"""
from __future__ import annotations

# Precision-curve thresholds (metres): success is evaluated at MANY thresholds from one rollout
# (the recorded continuous error), so we get a precision curve instead of one arbitrary cutoff.
THRESHOLDS = {
    "reach": [0.05, 0.03, 0.015],
    "grasp_lift": [0.06, 0.05, 0.03, 0.02],
    "place": [0.10, 0.06, 0.03, 0.015],
    "pick_place": [0.10, 0.06, 0.03, 0.015],
}

# The physical gates that must ALSO hold (beyond the precision threshold) for a real success.
# pick_place includes ``grasped`` so a placed-but-never-grasped composite cannot count as success.
GATE_SPEC = {
    "reach": [],
    "grasp_lift": ["lifted", "held", "upright", "stable"],
    "place": ["upright", "stable", "released"],
    "pick_place": ["grasped", "upright", "stable", "released"],
}


def success_at(error: float, gates: dict, thr: float) -> bool:
    """A trial is a success at precision ``thr`` iff error < thr AND all physical gates hold."""
    return bool(error < thr and all(gates.values()))
