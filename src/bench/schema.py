"""Task-bundle schema for the closed-loop task-success benchmark.

A task *bundle* is a self-contained directory describing one benchmark task, split into what the
model may see and what only the evaluator sees (docs/experiments/closed_loop_success_plan.md):

    tasks/<task>/<object>/<task_id>/
        meta.json          task, object, camera, success_spec, seed, start_grasped, ...
        start.png          observation at t0 (planner input)
        goal.png           goal image (planner target); goal_1.png, goal_2.png for multi-stage
        arrays.npz         qpos0/qvel0, qpos_start, qpos_goal(_1/_2), object_pose,
                           goal_object, grasp_pos, goal_ee, zone, ...
        contact_sheet.png  visual check artifact (optional)

The planner reads RGB goal images and restored qpos; the evaluator reads privileged arrays plus the
live simulator. Kept dependency-light (numpy + imageio) so bundles can be created/inspected without
the world model.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import imageio.v2 as imageio
import numpy as np


@dataclass
class TaskBundle:
    """One benchmark task. ``meta`` holds JSON-serialisable metadata (task, object, camera,
    image_hw, units, success_spec, seed, start_grasped, ...);
    ``images`` maps a name (``start``, ``goal``, ``goal_1``, ...) to an HxWx3 uint8 array;
    ``arrays`` maps a name to a numeric array; ``model_xml`` is optional for external tasks."""

    meta: dict
    images: dict[str, np.ndarray] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    model_xml: str | None = None

    @property
    def task_id(self) -> str:
        return str(self.meta["task_id"])

    def save(self, root: str) -> str:
        """Write the bundle to ``root/<task_id>/`` and return that directory."""
        out = os.path.join(root, self.task_id)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)
        for name, img in self.images.items():
            imageio.imwrite(os.path.join(out, f"{name}.png"), np.ascontiguousarray(img))
        if self.arrays:
            np.savez_compressed(os.path.join(out, "arrays.npz"), **self.arrays)
        if self.model_xml is not None:
            with open(os.path.join(out, "model.xml"), "w", encoding="utf-8") as f:
                f.write(self.model_xml)
        return out

    @classmethod
    def load(cls, path: str) -> "TaskBundle":
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        images: dict[str, np.ndarray] = {}
        for fn in os.listdir(path):
            if fn.endswith(".png") and fn != "contact_sheet.png":
                images[os.path.splitext(fn)[0]] = imageio.imread(os.path.join(path, fn))
        arrays: dict[str, np.ndarray] = {}
        npz = os.path.join(path, "arrays.npz")
        if os.path.exists(npz):
            with np.load(npz) as data:
                arrays = {k: data[k] for k in data.files}
        model_xml = None
        mx = os.path.join(path, "model.xml")
        if os.path.exists(mx):
            with open(mx, encoding="utf-8") as f:
                model_xml = f.read()
        return cls(meta=meta, images=images, arrays=arrays, model_xml=model_xml)


# Default success thresholds per task type (metres / radians / m/s). Difficulty tightens these;
# see docs/experiments/closed_loop_success_plan.md Section 4.
SUCCESS_DEFAULTS = {
    "reach": {"tau_reach": 0.05},
    "touch": {"move_tol": 0.03, "proximity": 0.02},
    "grasp_lift": {"lift_dz": 0.04, "grasp_radius": 0.05, "tilt_max_deg": 30.0, "v_settle": 0.05},
    "place": {"zone_radius": 0.06, "tilt_max_deg": 25.0, "v_settle": 0.05},
}
