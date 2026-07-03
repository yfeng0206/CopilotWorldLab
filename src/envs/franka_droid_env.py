"""DROID-style Franka + Robotiq 2F-85 reproduction environment.

Wraps the composed Franka Panda + Robotiq 2F-85 model (``franka_build``) with the same
interface ``MujocoPilotEnv`` exposes, but backed by a real 7-DoF arm driven in
end-effector space via differential IK. This is the paper-faithful substrate: an
exocentric camera, a 7-D EE state/action, and a Robotiq gripper matching DROID.

- ``render``              -> uint8 RGB from the exocentric camera (the observation)
- ``get_ee_state``        -> [x, y, z, roll, pitch, yaw, gripper]  (extrinsic-XYZ Euler,
                             gripper = MEASURED opening in [0, 1] from the driver joint)
- ``apply_action``        -> apply a 7-D EE delta dynamically (IK -> data.ctrl -> mj_step
                             for ~0.25 s, so the servos and physics actually move the arm)
- ``capture_goal_image``  -> render at a hypothetical EE pose without disturbing state

``apply_action`` is the real control path: it runs the position servos and physics, so
contacts, gravity and the gripper are physically consistent (grasping is possible). The
action translation is bounded to ~13 cm (the paper's per-action limit) and an action whose
IK solution is unreachable is rejected (the arm holds). ``set_ee_pose`` /
``capture_goal_image`` are an explicit KINEMATIC preview (teleport, no dynamics) used only
to render hypothetical goal poses; they never drive the real control state.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.envs.franka_build import (
    ARM_HOME_QPOS,
    EE_SITE,
    GRIPPER_ACTUATOR,
    GRIPPER_DRIVER_JOINT,
    GRIPPER_DRIVER_RANGE,
    build_franka_robotiq,
)
from src.utils import geometry as geo
from src.utils import ik as ik_solver

STATE_DIM = 7
ARM_DOF = 7
GRIPPER_CTRL_MAX = 255.0  # Robotiq 2F-85: ctrl 0 = open, 255 = closed
CONTROL_SUBSTEPS = 125    # ~0.25 s at dt=0.002: one action == one paper control step (4 fps)
MAX_TRANSLATION = 0.13    # metres per action (paper constrains actions to ~13 cm)
MAX_ROTATION = 0.5        # radians per action (keep orientation deltas bounded)
IK_FAIL_TOL = 0.02        # metres: reject an action whose translation IK residual exceeds this
IK_ROT_FAIL_TOL = 0.15    # radians (~8.6 deg): reject an action whose orientation IK residual exceeds this
GRIPPER_SETTLE_STEPS = 100  # physics steps to reach a commanded gripper opening in goal previews


class FrankaDroidEnv:
    def __init__(
        self,
        menagerie_dir: Optional[str] = None,
        render_width: int = 256,
        render_height: int = 256,
        default_camera: str = "exo_cam",
        ik_iters: int = 100,
        control_substeps: int = CONTROL_SUBSTEPS,
        max_translation: float = MAX_TRANSLATION,
        max_rotation: float = MAX_ROTATION,
        ik_fail_tol: float = IK_FAIL_TOL,
        ik_rot_fail_tol: float = IK_ROT_FAIL_TOL,
        seed: int = 0,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = (
            build_franka_robotiq(menagerie_dir) if menagerie_dir
            else build_franka_robotiq()
        )
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)  # scratch for IK forward-kinematics

        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.default_camera = default_camera
        self.ik_iters = int(ik_iters)
        self.control_substeps = int(control_substeps)
        self.max_translation = float(max_translation)
        self.max_rotation = float(max_rotation)
        self.ik_fail_tol = float(ik_fail_tol)
        self.ik_rot_fail_tol = float(ik_rot_fail_tol)
        self.rng = np.random.default_rng(seed)

        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if self._site < 0:
            raise ValueError(f"EE site '{EE_SITE}' not found")
        self._grip_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR
        )
        grip_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_DRIVER_JOINT)
        if grip_jid < 0:
            raise ValueError(f"gripper driver joint '{GRIPPER_DRIVER_JOINT}' not found")
        self._grip_qadr = int(self.model.jnt_qposadr[grip_jid])

        self._gripper_cmd = 0.0  # commanded gripper in [0, 1] (0 = open, 1 = closed)
        self.last_ik_pos_err = 0.0
        self.last_ik_rot_err = 0.0
        self.last_action_ok = True
        self._renderer = None
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> dict:
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:ARM_DOF] = ARM_HOME_QPOS
        self.data.ctrl[:ARM_DOF] = ARM_HOME_QPOS  # position servos hold home
        self._set_gripper_cmd(0.0)
        self.last_action_ok = True
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def _ee_pos_quat(self):
        pos = self.data.site_xpos[self._site].copy()
        quat = np.zeros(4)
        self._mujoco.mju_mat2Quat(quat, self.data.site_xmat[self._site])
        return pos, quat

    def _measured_gripper(self) -> float:
        """Measured opening in [0, 1] (0 = open, 1 = closed) from the driver joint qpos."""
        return float(np.clip(self.data.qpos[self._grip_qadr] / GRIPPER_DRIVER_RANGE, 0.0, 1.0))

    def get_ee_state(self) -> np.ndarray:
        pos, quat = self._ee_pos_quat()
        euler = geo.quat_to_euler_xyz(quat)
        return np.concatenate([pos, euler, [self._measured_gripper()]]).astype(np.float32)

    def _set_gripper_cmd(self, value: float) -> None:
        self._gripper_cmd = float(np.clip(value, 0.0, 1.0))
        self.data.ctrl[self._grip_act] = self._gripper_cmd * GRIPPER_CTRL_MAX

    def _drive_dynamic(self, target_pos, target_quat, gripper_delta=0.0, substeps=None) -> tuple:
        """Real control path: IK -> joint targets in data.ctrl -> step the physics.

        The 7-D action is atomic: if the target is unreachable (translation residual
        > ``ik_fail_tol`` or orientation residual > ``ik_rot_fail_tol``) the whole action is
        rejected -- neither the arm joint targets nor the gripper command are updated, and
        the arm holds while the sim steps.
        """
        substeps = self.control_substeps if substeps is None else int(substeps)
        q, pos_err, rot_err = ik_solver.solve_ik(
            self.model, self._ik_data, self._site, target_pos, target_quat,
            q_init=self.data.qpos[:ARM_DOF], arm_dof=ARM_DOF, iters=self.ik_iters,
        )
        self.last_ik_pos_err, self.last_ik_rot_err = pos_err, rot_err
        self.last_action_ok = pos_err <= self.ik_fail_tol and rot_err <= self.ik_rot_fail_tol
        if self.last_action_ok:
            self.data.ctrl[:ARM_DOF] = q
            self._set_gripper_cmd(self._gripper_cmd + gripper_delta)
        for _ in range(substeps):
            self._mujoco.mj_step(self.model, self.data)
        return pos_err, rot_err

    def _place_ee_kinematic(self, target_pos, target_quat) -> tuple:
        """KINEMATIC preview only: teleport the arm to the IK solution (no dynamics)."""
        q, pos_err, rot_err = ik_solver.solve_ik(
            self.model, self._ik_data, self._site, target_pos, target_quat,
            q_init=self.data.qpos[:ARM_DOF], arm_dof=ARM_DOF, iters=self.ik_iters,
        )
        self.data.qpos[:ARM_DOF] = q
        self.data.ctrl[:ARM_DOF] = q
        self._mujoco.mj_forward(self.model, self.data)
        return pos_err, rot_err

    def set_ee_pose(self, pos=None, euler=None, quat=None, gripper=None) -> tuple:
        cur_pos, cur_quat = self._ee_pos_quat()
        target_pos = cur_pos if pos is None else np.asarray(pos, dtype=np.float64).reshape(3)
        if quat is None:
            target_quat = cur_quat if euler is None else geo.euler_xyz_to_quat(
                *np.asarray(euler, dtype=np.float64).reshape(3))
        else:
            target_quat = geo.quat_normalize(quat)
        if gripper is not None:
            self._set_gripper_cmd(gripper)
        return self._place_ee_kinematic(target_pos, target_quat)

    def apply_action(self, delta, frame: str = "world") -> np.ndarray:
        """Apply a 7-D EE delta ``[dx,dy,dz, dR,dP,dY, dgrip]`` dynamically; return state.

        Translation is clamped to ``max_translation`` (by L2 norm) and rotation to
        ``max_rotation`` per action. The arm is driven by the position servos over
        ``control_substeps`` physics steps, so the result is physically consistent (contacts,
        gravity, grasp). The action is atomic: if the target is unreachable
        (``last_action_ok`` is False, i.e. ``last_ik_pos_err > ik_fail_tol`` or
        ``last_ik_rot_err > ik_rot_fail_tol``) neither the arm nor the gripper move.
        """
        delta = np.asarray(delta, dtype=np.float64).reshape(STATE_DIM)
        # Local convention: bound the L2 norm of the translation. The reference V-JEPA CEM
        # instead clips each xyz axis independently (a box); reconcile at interface calibration.
        trans = delta[:3].copy()
        t_norm = float(np.linalg.norm(trans))
        if t_norm > self.max_translation:
            trans *= self.max_translation / t_norm
        rot = delta[3:6].copy()
        r_norm = float(np.linalg.norm(rot))
        if r_norm > self.max_rotation:
            rot *= self.max_rotation / r_norm

        cur_pos, cur_quat = self._ee_pos_quat()
        target_pos = cur_pos + trans
        dquat = geo.euler_xyz_to_quat(*rot)
        target_quat = (geo.quat_mul(dquat, cur_quat) if frame == "world"
                       else geo.quat_mul(cur_quat, dquat))
        self._drive_dynamic(target_pos, geo.quat_normalize(target_quat), gripper_delta=delta[6])
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
                           camera: Optional[str] = None,
                           settle_steps: int = GRIPPER_SETTLE_STEPS) -> np.ndarray:
        """Render a hypothetical goal pose without disturbing the live state.

        The arm is teleported kinematically, but the gripper is a driven linkage whose
        finger pose is not set by ``mj_forward`` alone, so when a gripper opening is
        requested the physics is stepped briefly (arm held by its servos) to let the fingers
        reach the commanded opening -- otherwise open vs closed goals would render alike.
        """
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()
        saved_grip = self._gripper_cmd
        try:
            self.set_ee_pose(pos=pos, euler=euler, gripper=gripper)
            if gripper is not None and settle_steps:
                self.data.qvel[:] = 0.0
                for _ in range(int(settle_steps)):
                    self._mujoco.mj_step(self.model, self.data)
            return self.render(camera=camera)
        finally:
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self.data.ctrl[:] = saved_ctrl
            self._gripper_cmd = saved_grip
            self._mujoco.mj_forward(self.model, self.data)

    def get_observation(self, camera: Optional[str] = None) -> dict:
        return {"image": self.render(camera=camera), "ee_state": self.get_ee_state()}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self) -> "FrankaDroidEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
