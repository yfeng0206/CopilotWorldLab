"""MuJoCo step-1 pilot environment.

A thin wrapper around a minimal tabletop scene that exposes exactly the interface
the V-JEPA 2-AC world model will need later:

- ``render(...)``            -> uint8 RGB frame from a named camera (the observation)
- ``get_ee_state()``         -> 7-D end-effector state [x, y, z, roll, pitch, yaw, gripper]
- ``apply_action(delta7)``   -> apply a 7-D end-effector delta (matches V-JEPA 2-AC actions)
- ``capture_goal_image(...)``-> render the scene at a hypothetical EE pose (goal image)

The end-effector is a *kinematic* mocap body, so step 1 needs no arm, no IK and no
actuators. The only dynamic body is the vial. No world model is imported or run
here; this module is pure simulation + rendering.

Rendering requires an OpenGL context. On Windows that is the WGL backend (the only
option; EGL/OSMesa are Linux-only), which needs an interactive desktop session.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from src.utils import geometry as geo

# Full 7-D end-effector state / action layout, documented once.
STATE_DIM = 7  # [x, y, z, roll, pitch, yaw, gripper]


class MujocoPilotEnv:
    def __init__(
        self,
        model_path: str,
        render_width: int = 256,
        render_height: int = 256,
        default_camera: str = "scene_cam",
        ee_body: str = "ee",
        n_substeps: int = 5,
        gl_backend: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        # MUJOCO_GL must be set before mujoco's GL module is imported, so set it first.
        if gl_backend:
            os.environ.setdefault("MUJOCO_GL", gl_backend)
        # Import mujoco lazily so that importing this module never requires GL.
        import mujoco

        self._mujoco = mujoco

        self.model_path = os.path.abspath(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.default_camera = default_camera
        self.n_substeps = int(n_substeps)
        self.rng = np.random.default_rng(seed)

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if body_id < 0:
            raise ValueError(f"end-effector body '{ee_body}' not found in {self.model_path}")
        self._ee_mocap = int(self.model.body_mocapid[body_id])
        if self._ee_mocap < 0:
            raise ValueError(f"body '{ee_body}' is not a mocap body")

        self._gripper = 0.0
        self._renderer = None
        mujoco.mj_forward(self.model, self.data)
        # Remember the home pose so reset() restores it exactly.
        self._home_pos = self.data.mocap_pos[self._ee_mocap].copy()
        self._home_quat = self.data.mocap_quat[self._ee_mocap].copy()

    # ------------------------------------------------------------------ state
    def reset(self) -> dict:
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.mocap_pos[self._ee_mocap] = self._home_pos
        self.data.mocap_quat[self._ee_mocap] = self._home_quat
        self._gripper = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_ee_state(self) -> np.ndarray:
        pos = self.data.mocap_pos[self._ee_mocap].copy()
        euler = geo.quat_to_euler_xyz(self.data.mocap_quat[self._ee_mocap].copy())
        return np.concatenate([pos, euler, [self._gripper]]).astype(np.float32)

    def set_ee_pose(self, pos=None, euler=None, quat=None, gripper=None) -> None:
        if pos is not None:
            self.data.mocap_pos[self._ee_mocap] = np.asarray(pos, dtype=np.float64).reshape(3)
        if quat is None and euler is not None:
            quat = geo.euler_xyz_to_quat(*np.asarray(euler, dtype=np.float64).reshape(3))
        if quat is not None:
            self.data.mocap_quat[self._ee_mocap] = geo.quat_normalize(quat)
        if gripper is not None:
            self._gripper = float(np.clip(gripper, 0.0, 1.0))
        self._mujoco.mj_forward(self.model, self.data)

    def apply_action(self, delta, step_physics: bool = True, frame: str = "world") -> np.ndarray:
        """Apply a 7-D end-effector delta and return the new EE state.

        ``delta = [dx, dy, dz, droll, dpitch, dyaw, dgripper]``. Position is a
        world-frame translation; orientation is composed in the world frame
        (``frame='world'``) or the body frame (``frame='body'``); gripper is a
        signed increment clamped to ``[0, 1]``.
        """
        delta = np.asarray(delta, dtype=np.float64).reshape(STATE_DIM)
        pos = self.data.mocap_pos[self._ee_mocap].copy() + delta[:3]
        dquat = geo.euler_xyz_to_quat(*delta[3:6])
        cur_quat = self.data.mocap_quat[self._ee_mocap].copy()
        new_quat = geo.quat_mul(dquat, cur_quat) if frame == "world" else geo.quat_mul(cur_quat, dquat)

        self.data.mocap_pos[self._ee_mocap] = pos
        self.data.mocap_quat[self._ee_mocap] = geo.quat_normalize(new_quat)
        self._gripper = float(np.clip(self._gripper + delta[6], 0.0, 1.0))

        if step_physics:
            for _ in range(self.n_substeps):
                self._mujoco.mj_step(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_ee_state()

    # --------------------------------------------------------------- rendering
    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        return self._renderer

    def render(self, camera: Optional[str] = None, width: Optional[int] = None,
               height: Optional[int] = None) -> np.ndarray:
        """Return a ``uint8`` ``[H, W, 3]`` RGB frame from ``camera``."""
        cam = camera or self.default_camera
        w = int(width or self.render_width)
        h = int(height or self.render_height)
        if w == self.render_width and h == self.render_height:
            renderer = self._ensure_renderer()
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()
        renderer = self._mujoco.Renderer(self.model, height=h, width=w)
        try:
            renderer.update_scene(self.data, camera=cam)
            return renderer.render()
        finally:
            renderer.close()

    def capture_goal_image(self, pos=None, euler=None, gripper=None,
                           camera: Optional[str] = None) -> np.ndarray:
        """Render the scene at a hypothetical EE pose without disturbing state."""
        saved = (
            self.data.mocap_pos[self._ee_mocap].copy(),
            self.data.mocap_quat[self._ee_mocap].copy(),
            self._gripper,
        )
        try:
            self.set_ee_pose(pos=pos, euler=euler, gripper=gripper)
            return self.render(camera=camera)
        finally:
            self.data.mocap_pos[self._ee_mocap] = saved[0]
            self.data.mocap_quat[self._ee_mocap] = saved[1]
            self._gripper = saved[2]
            self._mujoco.mj_forward(self.model, self.data)

    def get_observation(self, camera: Optional[str] = None) -> dict:
        return {"image": self.render(camera=camera), "ee_state": self.get_ee_state()}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "MujocoPilotEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
