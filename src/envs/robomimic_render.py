"""Render robomimic *raw* demonstrations on Windows without the robosuite runtime.

robosuite's env stepping is blocked on this Windows setup (mujoco 3.10 ``mj_fullM``,
lessons_learned #11), but a demo's saved MuJoCo state can be *rendered* with plain MuJoCo: build
the model from the demo's embedded ``model_file`` XML, set the flattened state, ``mj_forward``
(never ``mj_step`` -- dynamics stepping is what crashes), and render offscreen. This module
provides the asset-path patch and a small renderer that also exposes privileged truth (EE site
pose, object body pose) for building benchmark task bundles.

Verified cameras in the Lift model: ``frontview, birdview, agentview, sideview,
robot0_robotview, robot0_eye_in_hand``.
"""
from __future__ import annotations

import os
import re

import numpy as np

from src.utils.geometry import quat_to_euler_xyz


def local_robosuite_assets() -> str:
    import robosuite

    return os.path.join(os.path.dirname(robosuite.__file__), "models", "assets").replace("\\", "/")


def patch_asset_paths(xml: str, assets_dir: str | None = None) -> str:
    """Rewrite absolute asset paths in a robosuite ``model_file`` to the local install.

    robomimic demos embed the collector's absolute paths (e.g.
    ``/home/.../robosuite/models/assets/...``). This replicates robosuite ``edit_model_xml``'s
    path fix without instantiating an env: any path ending in ``robosuite/models/assets/`` is
    rewritten to point at the local package's asset directory.
    """
    assets = (assets_dir or local_robosuite_assets()).replace("\\", "/")
    return re.sub(r'[^"\s>]*?robosuite/models/assets/', assets + "/", xml)


def _names(model, objtype) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, objtype, i) for i in range(_count(model, objtype))]


def _count(model, objtype) -> int:
    import mujoco

    return {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_SITE: model.nsite,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        mujoco.mjtObj.mjOBJ_CAMERA: model.ncam,
    }[objtype]


class RobomimicDemoRenderer:
    """Load a robomimic HDF5 and render/inspect its demos frame-by-frame."""

    def __init__(self, hdf5_path: str, height: int = 256, width: int = 256):
        import h5py

        self.hdf5_path = hdf5_path
        self.height, self.width = height, width
        self._f = h5py.File(hdf5_path, "r")
        self._data = self._f["data"]
        self.demo_names = sorted(self._data.keys(), key=lambda s: int(s.split("_")[-1]))
        self._model = None
        self._data_mj = None
        self._renderer = None
        self._states = None
        self._actions = None
        self.eef_site = None
        self.object_body = None
        self.finger_joints: list[int] = []

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def cameras(self) -> list[str]:
        import mujoco

        return _names(self._model, mujoco.mjtObj.mjOBJ_CAMERA)

    def load_demo(self, demo_name: str) -> None:
        import mujoco

        demo = self._data[demo_name]
        xml = patch_asset_paths(demo.attrs["model_file"])
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data_mj = mujoco.MjData(self._model)
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        self._states = np.asarray(demo["states"][:], dtype=np.float64)
        self._actions = np.asarray(demo["actions"][:], dtype=np.float32)
        self.model_xml = xml
        self._resolve_named_entities()

    def _resolve_named_entities(self) -> None:
        import mujoco

        m = self._model
        sites = _names(m, mujoco.mjtObj.mjOBJ_SITE)
        grip = [s for s in sites if s and s.endswith("grip_site") and not s.endswith("cylinder")]
        self.eef_site = grip[0] if grip else next((s for s in sites if s and "grip" in s), None)
        # object = body carrying the (single) free joint that is not the robot base
        free_bodies = [int(m.jnt_bodyid[i]) for i in range(m.njnt)
                       if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        self.object_body = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, free_bodies[0])
                            if free_bodies else None)
        self.finger_joints = [i for i in range(m.njnt)
                              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                              .find("finger_joint") >= 0]

    @property
    def num_frames(self) -> int:
        return 0 if self._states is None else int(self._states.shape[0])

    @property
    def actions(self) -> np.ndarray:
        return self._actions

    def set_frame(self, t: int) -> None:
        import mujoco

        m, d = self._model, self._data_mj
        s = self._states[t]
        d.time = float(s[0])
        d.qpos[:] = s[1:1 + m.nq]
        d.qvel[:] = s[1 + m.nq:1 + m.nq + m.nv]
        mujoco.mj_forward(m, d)

    def render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self._data_mj, camera=camera)
        return self._renderer.render()

    def gripper_width(self) -> float:
        if not self.finger_joints:
            return 0.0
        qadr = [int(self._model.jnt_qposadr[j]) for j in self.finger_joints]
        return float(sum(self._data_mj.qpos[a] for a in qadr))

    def ee_state(self) -> np.ndarray:
        """End-effector pose as [x, y, z, roll, pitch, yaw, gripper] (extrinsic XYZ euler)."""
        import mujoco

        m, d = self._model, self._data_mj
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.eef_site)
        pos = np.asarray(d.site_xpos[sid], dtype=np.float64)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(d.site_xmat[sid], dtype=np.float64))
        euler = quat_to_euler_xyz(quat)
        return np.concatenate([pos, euler, [self.gripper_width()]]).astype(np.float32)

    def object_state(self) -> np.ndarray:
        """Object pose as [x, y, z, qw, qx, qy, qz] from the manipuland body."""
        import mujoco

        if self.object_body is None:
            return np.full(7, np.nan, dtype=np.float32)
        m, d = self._model, self._data_mj
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.object_body)
        return np.concatenate([np.asarray(d.xpos[bid]),
                               np.asarray(d.xquat[bid])]).astype(np.float32)

    def flattened_state(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (qpos, qvel) at frame ``t`` for resetting a steppable env later."""
        m = self._model
        s = self._states[t]
        return (s[1:1 + m.nq].astype(np.float64).copy(),
                s[1 + m.nq:1 + m.nq + m.nv].astype(np.float64).copy())
