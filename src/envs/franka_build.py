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

import math
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
# closed_loop_success_plan.md). A single free-joint manipuland body (CUBE_BODY) carries one of the
# object geometries below; the env identifies it by body, so all object types share the accessors.
CUBE_BODY = "cube"         # manipuland body name (shared across object types; env looks it up by body)
CUBE_GEOM = "cube_geom"
CUBE_HALF = 0.02            # 4 cm cube
CUBE_START = (0.5, -0.10, 0.24)   # on the table top (z = 0.22 + half)

# Box: a rigid rectangular block grasped across its narrow width (paper's Box object, arXiv
# 2506.09985 Fig. 14) -- needs precise finger width. 5 x 4 x 6 cm (half-sizes below).
BOX_HALF = (0.025, 0.02, 0.03)

# Cup: a procedural thin-wall OPEN cylinder (MuJoCo has no hollow-cylinder primitive), grasped by the
# rim -- one finger inside, fingers around the wall (paper's Cup object). Built as a ring of box wall
# segments + a bottom disk, all in the one manipuland body.
CUP_OUTER_R = 0.03         # outer radius (6 cm outer diameter, within the 2F-85 ~85 mm stroke)
CUP_WALL_T = 0.006         # wall thickness
CUP_HALF_H = 0.035         # 7 cm tall
CUP_SEGMENTS = 12          # wall box segments around the ring
CUP_BOTTOM_HALF = 0.004    # bottom disk half-thickness

# Per-object rest half-height (for placing the object resting on the table), grasp style, the grasp
# z offset from the object CENTRE, and an xy grasp offset. cube/box grasp top-down at the centre; the
# cup is grasped at the RIM -- the TCP is offset along the gripper closing axis (world Y) by ~one wall
# radius so one finger goes inside the hollow and one outside, gripping the rim (paper's cup grasp).
OBJECT_SPECS = {
    "cube": {"rest_half_z": CUBE_HALF, "grasp": "top", "grasp_dz": 0.005, "grasp_off": (0.0, 0.0)},
    "box": {"rest_half_z": BOX_HALF[2], "grasp": "top", "grasp_dz": 0.0, "grasp_off": (0.0, 0.0)},
    "cup": {"rest_half_z": CUP_HALF_H, "grasp": "rim", "grasp_dz": 0.015,
            "grasp_off": (0.0, CUP_OUTER_R - CUP_WALL_T / 2.0)},
}

# Manipuland material: high tangential friction so the grip holds under static load; ~16-30 g so the
# grasp is reliable and non-slipping.
_OBJ_FRICTION = [2.0, 0.1, 0.01]
_OBJ_DENSITY = 250.0

# Static distractor clutter for visual realism (paper's cluttered table, Fig. 14). Visual-only
# (contype/conaffinity 0) and outside the reachable workspace, so they never affect physics.
DISTRACTORS = (
    ("distractor_box", "box", (0.72, -0.30, 0.24), (0.04, 0.03, 0.04), (0.25, 0.35, 0.80, 1.0)),
    ("distractor_can", "cylinder", (0.73, 0.30, 0.25), (0.03, 0.05, 0.0), (0.85, 0.45, 0.20, 1.0)),
    ("distractor_ball", "sphere", (0.30, 0.34, 0.25), (0.035, 0.0, 0.0), (0.55, 0.30, 0.65, 1.0)),
)

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



def _apply_object_material(g) -> None:
    """Common manipuland material: grip-friendly friction, contact dim, density."""
    g.friction = list(_OBJ_FRICTION)
    g.condim = 4
    g.density = _OBJ_DENSITY


def _add_object_geoms(body, object_type: str, mujoco) -> None:
    """Attach the geometry for ``object_type`` (cube | box | cup) to a free-joint manipuland body."""
    if object_type == "cube":
        g = body.add_geom()
        g.name = CUBE_GEOM
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [CUBE_HALF, CUBE_HALF, CUBE_HALF]
        g.rgba = [0.85, 0.15, 0.15, 1.0]
        _apply_object_material(g)
    elif object_type == "box":
        g = body.add_geom()
        g.name = "box_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(BOX_HALF)
        g.rgba = [0.20, 0.35, 0.80, 1.0]
        _apply_object_material(g)
    elif object_type == "cup":
        bottom = body.add_geom()
        bottom.name = "cup_bottom"
        bottom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        bottom.size = [CUP_OUTER_R, CUP_BOTTOM_HALF, 0.0]
        bottom.pos = [0.0, 0.0, -CUP_HALF_H + CUP_BOTTOM_HALF]
        bottom.rgba = [0.90, 0.55, 0.62, 1.0]
        _apply_object_material(bottom)
        r_mid = CUP_OUTER_R - CUP_WALL_T / 2.0
        seg_tan = math.pi * CUP_OUTER_R / CUP_SEGMENTS * 1.15  # half tangential length, slight overlap
        for i in range(CUP_SEGMENTS):
            th = 2.0 * math.pi * i / CUP_SEGMENTS
            g = body.add_geom()
            g.name = f"cup_wall_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [CUP_WALL_T / 2.0, seg_tan, CUP_HALF_H]
            g.pos = [r_mid * math.cos(th), r_mid * math.sin(th), 0.0]
            g.quat = [math.cos(th / 2.0), 0.0, 0.0, math.sin(th / 2.0)]  # rotate about z so local x is radial
            g.rgba = [0.90, 0.55, 0.62, 1.0]
            _apply_object_material(g)
    else:
        raise ValueError(f"unknown object_type '{object_type}' (expected cube | box | cup)")


def _add_distractors(world, mujoco) -> None:
    """Add static, visual-only clutter outside the workspace for paper-like scene realism."""
    type_map = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
    for name, gtype, pos, size, rgba in DISTRACTORS:
        body = world.add_body()
        body.name = name
        body.pos = list(pos)
        g = body.add_geom()
        g.name = f"{name}_geom"
        g.type = type_map[gtype]
        g.size = list(size)
        g.rgba = list(rgba)
        g.contype = 0       # visual-only: never collides, so it cannot perturb the task
        g.conaffinity = 0


def build_franka_robotiq(menagerie_dir: str = DEFAULT_MENAGERIE, add_camera: bool = True,
                         add_target: bool = False, add_object: bool = False,
                         add_zone: bool = False, object_type: str = "cube",
                         add_distractors: bool = False):
    """Return a compiled ``mujoco.MjModel`` of the Franka + Robotiq DROID-style scene.

    ``add_object`` adds a graspable free-joint manipuland on the table (``object_type`` selects
    ``cube`` | ``box`` | ``cup``) and ``add_zone`` a visual place-target marker -- the manipulands for
    the closed-loop grasp/place benchmark. ``add_distractors`` scatters static, visual-only clutter
    outside the workspace for paper-like scene realism.
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
        obj = world.add_body()
        obj.name = CUBE_BODY
        rest_half_z = OBJECT_SPECS[object_type]["rest_half_z"]
        obj.pos = [CUBE_START[0], CUBE_START[1], TABLE_TOP_Z + rest_half_z]
        obj.add_freejoint()
        _add_object_geoms(obj, object_type, mujoco)

    if add_distractors:
        _add_distractors(world, mujoco)

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
