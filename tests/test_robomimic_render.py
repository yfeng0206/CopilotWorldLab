"""Tests for the robomimic raw-demo renderer and the task-bundle schema (Stage 0)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.bench.schema import SUCCESS_DEFAULTS, TaskBundle
from src.envs.robomimic_render import patch_asset_paths

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIFT = os.path.join(_REPO_ROOT, "data", "robomimic", "v1.5", "lift", "ph", "demo_v15.hdf5")


def test_patch_asset_paths_rewrites_absolute_paths():
    xml = ('<mujoco><asset>'
           '<mesh file="/home/someone/code/robosuite/models/assets/robots/panda/link0.stl"/>'
           '</asset></mujoco>')
    out = patch_asset_paths(xml, assets_dir="C:/local/robosuite/models/assets")
    assert "/home/someone/code" not in out
    assert "C:/local/robosuite/models/assets/robots/panda/link0.stl" in out


def test_bundle_roundtrip(tmp_path):
    meta = {"task_id": "unit_demo", "task_type": "grasp_lift", "difficulty": "easy",
            "success_spec": {"type": "grasp_lift", **SUCCESS_DEFAULTS["grasp_lift"]}}
    img = (np.random.default_rng(0).integers(0, 255, (16, 16, 3))).astype(np.uint8)
    arrays = {"start_state": np.arange(7, dtype=np.float32),
              "object_state": np.arange(7, dtype=np.float32) + 1}
    saved = TaskBundle(meta=meta, images={"start": img, "goal": img},
                       arrays=arrays, model_xml="<mujoco/>").save(str(tmp_path))
    loaded = TaskBundle.load(saved)
    assert loaded.meta["task_id"] == "unit_demo"
    assert loaded.meta["task_type"] == "grasp_lift"
    np.testing.assert_array_equal(loaded.images["start"], img)
    np.testing.assert_array_equal(loaded.arrays["start_state"], arrays["start_state"])
    assert loaded.model_xml == "<mujoco/>"


@pytest.mark.skipif(not os.path.exists(_LIFT), reason="robomimic Lift raw demo not downloaded")
def test_render_lift_demo_object_lifts():
    from src.envs.robomimic_render import RobomimicDemoRenderer

    with RobomimicDemoRenderer(_LIFT, height=128, width=128) as rndr:
        rndr.load_demo(rndr.demo_names[0])
        assert rndr.object_body == "cube_main"
        assert "grip_site" in rndr.eef_site
        assert "agentview" in rndr.cameras
        rndr.set_frame(0)
        img0 = rndr.render("agentview")
        obj0 = rndr.object_state()
        assert img0.shape == (128, 128, 3) and img0.dtype == np.uint8
        rndr.set_frame(rndr.num_frames - 1)
        objg = rndr.object_state()
        # a Lift demo is a successful grasp-lift: the cube ends higher than it started
        assert float(objg[2] - obj0[2]) > 0.02
