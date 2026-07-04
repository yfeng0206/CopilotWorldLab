"""Render Franka end-effector transitions from a sweep of exocentric cameras (phase 1).

Produces the input for the zero-shot transfer test and the camera-placement ablation. For
each start pose and each canonical end-effector action, it drives the real Franka + Robotiq
arm one control step with ``FrankaDroidEnv`` (physics, not teleport) and renders the
before/after frames from every camera, saving them in the paper's trajectory format so
``scripts/energy_landscape_repro.py --traj`` can score the latent energy landscape on our own
simulator renders. Each ``.npz`` embeds ``camera``/``action``/``pose`` metadata so the scorer
aggregates per camera (not per transition).

The free cameras are parameterized by azimuth / elevation / distance about a fixed workspace
lookat, so placements can be swept without editing the model; the exact built-in ``exo_cam``
is also included as a fair, un-reparameterized reference. This is the knob the ablation
varies: V-JEPA 2-AC was trained only on DROID exocentric views, so a poor camera match is a
prime suspect if zero-shot transfer is weak. Every camera renders the SAME physical
transition, so the only variable between cameras is the viewpoint.

Uses this project's ``src`` only (no world model here); it must run in a separate process
from the energy analysis, which uses the vendored repo's colliding ``src``.

    python scripts/render_franka_transitions.py                       # full sweep
    python scripts/render_franka_transitions.py --step 0.06 --poses 3 --out outputs/transitions
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.franka_droid_env import FrankaDroidEnv  # noqa: E402

# Workspace point the free cameras aim at (table top, roughly where the arm works).
LOOKAT = (0.5, 0.0, 0.35)

# Free-camera placements to ablate: (azimuth_deg, elevation_deg, distance_m). Azimuth sweeps
# the viewing side, elevation the height; "top_down" mimics a wrist camera (out of
# distribution). "exo_named" (added separately) is the exact built-in exo_cam reference.
FREE_CAMERAS = {
    "az135_el20": (-135.0, -20.0, 1.5),   # over-the-shoulder, low
    "az135_el45": (-135.0, -45.0, 1.5),   # over-the-shoulder, high
    "az90_el20": (-90.0, -20.0, 1.5),     # side, low
    "az90_el45": (-90.0, -45.0, 1.5),     # side, high
    "az45_el20": (-45.0, -20.0, 1.5),     # opposite shoulder, low
    "az45_el45": (-45.0, -45.0, 1.5),     # opposite shoulder, high
    "top_down": (-90.0, -85.0, 1.4),      # near top-down (wrist-like, out of distribution)
}


def canonical_actions(step: float):
    """Six single-axis end-effector deltas (+/- x, y, z) spanning all three axes."""
    return {
        "px": [step, 0.0, 0.0, 0, 0, 0, 0.0], "nx": [-step, 0.0, 0.0, 0, 0, 0, 0.0],
        "py": [0.0, step, 0.0, 0, 0, 0, 0.0], "ny": [0.0, -step, 0.0, 0, 0, 0, 0.0],
        "pz": [0.0, 0.0, step, 0, 0, 0, 0.0], "nz": [0.0, 0.0, -step, 0, 0, 0, 0.0],
    }


def make_free_camera(mujoco, azimuth, elevation, distance, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    cam.distance = float(distance)
    cam.lookat[:] = np.asarray(lookat, dtype=np.float64)
    return cam


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join("outputs", "transitions"))
    parser.add_argument("--size", type=int, default=256, help="render width/height (px)")
    parser.add_argument("--step", type=float, default=0.06,
                        help="per-action EE delta magnitude (m); keep < the scorer grid half-extent")
    parser.add_argument("--poses", type=int, default=3, help="start poses per action")
    parser.add_argument("--reposition", type=float, default=0.06,
                        help="max random EE offset used to generate alternative start poses (m)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    actions = canonical_actions(args.step)

    env = FrankaDroidEnv(render_width=args.size, render_height=args.size)
    renderer = mujoco.Renderer(env.model, height=args.size, width=args.size)
    free_cams = {name: make_free_camera(mujoco, *p, LOOKAT) for name, p in FREE_CAMERAS.items()}
    cam_names = list(FREE_CAMERAS) + ["exo_named"]

    def render_from(cam_key):
        cam = "exo_cam" if cam_key == "exo_named" else free_cams[cam_key]
        renderer.update_scene(env.data, camera=cam)
        return renderer.render().copy()

    def goto_start(pose_idx):
        """Reset to home, then (for pose_idx > 0) settle into a random nearby start pose."""
        env.reset()
        if pose_idx == 0:
            return True
        offset = rng.uniform(-args.reposition, args.reposition, size=3)
        env.apply_action([offset[0], offset[1], offset[2], 0, 0, 0, 0.0])
        return bool(env.last_action_ok)

    n_written, n_skipped = 0, 0
    for pose_idx in range(args.poses):
        for act_name, delta in actions.items():
            if not goto_start(pose_idx):
                n_skipped += len(cam_names)
                continue
            state_a = env.get_ee_state().copy()
            frames_a = {c: render_from(c) for c in cam_names}
            env.apply_action(delta)
            if not env.last_action_ok:
                print(f"skip pose{pose_idx}/{act_name}: action rejected "
                      f"(ik_pos_err={env.last_ik_pos_err:.3f})", flush=True)
                n_skipped += len(cam_names)
                continue
            state_b = env.get_ee_state().copy()
            frames_b = {c: render_from(c) for c in cam_names}
            states = np.stack([state_a, state_b])[None].astype(np.float32)  # [1,2,7]
            moved = float(np.linalg.norm(state_b[:3] - state_a[:3]))
            for cam_key in cam_names:
                obs = np.stack([frames_a[cam_key], frames_b[cam_key]])[None].astype(np.uint8)
                dst = os.path.join(args.out, f"t_{cam_key}_{act_name}_p{pose_idx}.npz")
                np.savez(dst, observations=obs, states=states,
                         camera=cam_key, action=act_name, pose=pose_idx)
                n_written += 1
            print(f"pose{pose_idx}/{act_name}: |EE move|={moved:.3f} m -> {len(cam_names)} cameras",
                  flush=True)

    renderer.close()
    print(f"done: {n_written} transitions written, {n_skipped} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
