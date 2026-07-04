"""Compose the DROID-style reproduction model: Franka Panda + Robotiq 2F-85.

The paper trains V-JEPA 2-AC on DROID, whose hardware is a Franka Panda arm with a
Robotiq 2F-85 gripper viewed from a fixed exocentric camera. MuJoCo Menagerie ships
the arm and the gripper as separate models; this module composes them with the
``mjSpec`` editing API (attaching the gripper at the arm's ``attachment_site``) and
adds a table, a floor, a light and a fixed third-person camera.

Nothing here runs a world model. The model is built for the reproduction environment
(``FrankaDroidEnv``).

Fetch the Menagerie models first (into the gitignored ``third_party/``):

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
    cd third_party/mujoco_menagerie && git sparse-checkout set franka_emika_panda robotiq_2f85
"""
from __future__ import annotations

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MENAGERIE = os.path.join(_REPO_ROOT, "third_party", "mujoco_menagerie")

# The arm's tool flange exposes site "attachment_site"; the Robotiq root body is
# "base_mount". Gripper elements are prefixed "2f85_" in the composed model.
GRIPPER_PREFIX = "2f85_"
ATTACH_SITE = "attachment_site"  # arm tool flange: where the gripper is mounted
EE_SITE = "2f85_pinch"           # Robotiq TCP / grasp point: control + state reference
GRIPPER_ACTUATOR = "2f85_fingers_actuator"  # ctrl 0..255 (0 = open, 255 = closed)
GRIPPER_DRIVER_JOINT = "2f85_right_driver_joint"  # measured opening, qpos in [0, 0.8]
GRIPPER_DRIVER_RANGE = 0.8
ARM_HOME_QPOS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]

# Graspable object + place target for the closed-loop benchmark (docs/experiments/
# closed_loop_success_plan.md). A 4 cm cube is well within the Robotiq 2F-85's ~85 mm stroke.
CUBE_BODY = "cube"
CUBE_GEOM = "cube_geom"
CUBE_HALF = 0.02            # 4 cm cube
CUBE_START = (0.5, -0.10, 0.24)   # on the table top (z = 0.22 + half)
PLACE_ZONE_BODY = "place_zone"
PLACE_ZONE_CENTER = (0.5, 0.15, 0.221)
PLACE_ZONE_RADIUS = 0.05
TABLE_TOP_Z = 0.22

# Best zero-shot planning camera from the camera-placement ablation
# (docs/experiments/energy_landscape_and_camera_ablation.md): an opposite-shoulder, high
# exocentric free camera that needs almost no W* interface rotation (~8 deg residual).
PLANNING_CAMERA = {"azimuth": -45.0, "elevation": -45.0, "distance": 1.5,
                   "lookat": (0.5, 0.0, 0.35)}


def make_free_camera(azimuth: float, elevation: float, distance: float, lookat) -> "object":
    """Build a free ``mujoco.MjvCamera`` (azimuth/elevation/distance about a lookat point).

    Used for the camera-placement ablation and for the validated planning view
    (``PLANNING_CAMERA``); pass the returned camera to ``Renderer.update_scene``.
    """
    import mujoco
    import numpy as np

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    return cam



def build_franka_robotiq(menagerie_dir: str = DEFAULT_MENAGERIE, add_camera: bool = True,
                         add_target: bool = False, add_object: bool = False,
                         add_zone: bool = False):
    """Return a compiled ``mujoco.MjModel`` of the Franka + Robotiq DROID-style scene.

    ``add_object`` adds a graspable free-joint cube on the table and ``add_zone`` a visual
    place-target marker -- the manipulands for the closed-loop grasp/place benchmark.
    """
    import mujoco

    arm_path = os.path.join(menagerie_dir, "franka_emika_panda", "panda_nohand.xml")
    grip_path = os.path.join(menagerie_dir, "robotiq_2f85", "2f85.xml")
    for path in (arm_path, grip_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}. Fetch the MuJoCo Menagerie models first (see this "
                "module's docstring)."
            )

    arm = mujoco.MjSpec.from_file(arm_path)
    grip = mujoco.MjSpec.from_file(grip_path)

    # Mount the gripper on the arm's tool flange.
    arm.site(ATTACH_SITE).attach_body(grip.body("base_mount"), GRIPPER_PREFIX, "")

    # The Robotiq model is authored for an elliptic friction cone; the attach merge keeps
    # the arm's (pyramidal) option, so restore it explicitly for grip fidelity.
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10.0

    world = arm.worldbody

    floor = world.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.rgba = [0.30, 0.32, 0.36, 1.0]

    table = world.add_body()
    table.name = "table"
    table.pos = [0.5, 0.0, 0.20]
    top = table.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = [0.35, 0.5, 0.02]
    top.rgba = [0.75, 0.68, 0.55, 1.0]

    light = world.add_light()
    light.pos = [0.4, 0.0, 1.6]
    light.dir = [0.0, 0.0, -1.0]
    light.diffuse = [0.6, 0.6, 0.6]
    light.castshadow = True
    try:  # directional light if this build exposes the enum
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    except AttributeError:
        pass

    if add_camera:
        cam = world.add_camera()
        cam.name = "exo_cam"  # fixed third-person, DROID-like
        cam.pos = [1.15, -0.95, 0.85]
        cam.fovy = 45
        cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        cam.targetbody = "table"

    if add_target:
        target = world.add_body()
        target.name = "target"
        target.mocap = True
        target.pos = [0.5, 0.0, 0.45]
        marker = target.add_geom()
        marker.name = "target_marker"
        marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
        marker.size = [0.02, 0.0, 0.0]
        marker.rgba = [0.9, 0.2, 0.2, 0.55]
        marker.contype = 0
        marker.conaffinity = 0
        marker.group = 1

    if add_object:
        cube = world.add_body()
        cube.name = CUBE_BODY
        cube.pos = list(CUBE_START)
        cube.add_freejoint()
        g = cube.add_geom()
        g.name = CUBE_GEOM
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [CUBE_HALF, CUBE_HALF, CUBE_HALF]
        g.rgba = [0.85, 0.15, 0.15, 1.0]
        g.friction = [2.0, 0.1, 0.01]  # high tangential friction so the grip holds under static load
        g.condim = 4
        g.density = 250.0  # ~16 g for a 4 cm cube: light enough for a reliable, non-slipping grasp

    if add_zone:
        zone = world.add_body()
        zone.name = PLACE_ZONE_BODY
        zone.pos = list(PLACE_ZONE_CENTER)
        marker = zone.add_geom()
        marker.name = "place_zone_marker"
        marker.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        marker.size = [PLACE_ZONE_RADIUS, 0.001, 0.0]
        marker.rgba = [0.2, 0.7, 0.2, 0.40]
        marker.contype = 0
        marker.conaffinity = 0
        marker.group = 1

    return arm.compile()
