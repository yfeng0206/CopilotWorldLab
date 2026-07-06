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
    # fixed-bundle task set (swept sphere radius x, metres, tight -> loose)
    "grasp": [0.06, 0.03, 0.02],
    "reach_with_object": [0.10, 0.06, 0.03, 0.015],
    "grasp_and_reach": [0.10, 0.06, 0.03, 0.015],
}

# The physical gates that must ALSO hold (beyond the precision threshold) for a real success.
# pick_place includes ``grasped`` so a placed-but-never-grasped composite cannot count as success.
GATE_SPEC = {
    "reach": [],
    "grasp_lift": ["lifted", "held", "upright", "stable"],
    "place": ["upright", "stable", "released"],
    "pick_place": ["grasped", "upright", "stable", "released"],
    # fixed-bundle task set
    "grasp": ["lifted", "held", "upright", "stable"],
    "reach_with_object": ["held", "upright"],
    "grasp_and_reach": ["held", "upright"],
}


def success_at(error: float, gates: dict, thr: float) -> bool:
    """A trial is a success at precision ``thr`` iff error < thr AND all physical gates hold."""
    return bool(error < thr and all(gates.values()))


# Canonical task-name registries (names only; the runner binds LEGACY_TASKS to scripted functions).
# ``pick_place`` deliberately appears in BOTH: it has a legacy random-scenario implementation and a
# fixed-bundle implementation.
LEGACY_TASKS = ["reach", "grasp_lift", "place", "pick_place"]
BUNDLE_TASKS = ["grasp", "reach_with_object", "grasp_and_reach", "pick_place"]


def validate_task_mode(tasks, bundles, legacy=LEGACY_TASKS, bundle_tasks=BUNDLE_TASKS):
    """Return an error message if the (``tasks``, ``--bundles``) combination is invalid, else None.

    Fixed-bundle-only tasks (grasp / reach_with_object / grasp_and_reach) require ``--bundles``;
    legacy random tasks (reach / grasp_lift / place) cannot run under ``--bundles``. ``pick_place``
    is valid in either mode. Pure, so it is unit-testable without argparse or the model."""
    bundle_only = [t for t in bundle_tasks if t not in legacy]
    if bundles:
        stray = [t for t in tasks if t not in bundle_tasks]
        if stray:
            return (f"--bundles supports only {list(bundle_tasks)}; got unsupported {stray}. "
                    f"Drop them or run without --bundles for the legacy tasks {list(legacy)}.")
    else:
        need = [t for t in tasks if t in bundle_only]
        if need:
            return (f"tasks {need} are fixed-bundle only and require --bundles <dir> "
                    f"(e.g. --bundles tasks). Legacy tasks are {list(legacy)}.")
    return None
