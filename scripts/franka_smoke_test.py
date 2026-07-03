"""Franka Emika Panda smoke test in MuJoCo (no world model).

Loads the official MuJoCo Menagerie Panda, prints its structure, renders it,
actuates a joint to confirm the end-effector moves, and measures sim/render
timing against the V-JEPA 2 paper's control cadence (4 fps, 0.25 s per action,
16-frame / 4 s clips at 256x256).

This is a pure-simulation sanity check: it imports no world model and runs no
inference. It answers "does a real Franka load, render, actuate, and is the
simulation cheap relative to the paper's ~16 s/action model budget?".

Fetch the model first (into the gitignored third_party/):

    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
    cd third_party/mujoco_menagerie && git sparse-checkout set franka_emika_panda

Run from the repository root:

    python scripts/franka_smoke_test.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

DEFAULT_SCENE = os.path.join(
    "third_party", "mujoco_menagerie", "franka_emika_panda", "scene.xml"
)
# V-JEPA 2-AC control cadence (arXiv:2506.09985, Section 3.1).
ACTION_FPS = 4
CLIP_FRAMES = 16
RENDER_HW = 256


def _names(model, objtype, count):
    import mujoco

    out = []
    for i in range(count):
        out.append(mujoco.mj_id2name(model, objtype, i))
    return out


def summarize(model, data) -> None:
    import mujoco

    print("=== model summary ===")
    print(f"  timestep dt      : {model.opt.timestep} s")
    print(f"  nq/nv/nu/nbody   : {model.nq}/{model.nv}/{model.nu}/{model.nbody}")
    print(f"  joints ({model.njnt}): {_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)}")
    print(f"  actuators ({model.nu}): {_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)}")
    print(f"  cameras ({model.ncam}): {_names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)}")
    print("  actuator ctrl ranges:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"    {name:12s}  [{lo:.4f}, {hi:.4f}]")


def go_home(model, data) -> None:
    import mujoco

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
        data.ctrl[:] = model.key_ctrl[key]  # hold the home pose with the position servos
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def main() -> None:
    import mujoco

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--save", default=os.path.join("outputs", "franka_home.png"))
    parser.add_argument("--steps", type=int, default=2000, help="steps for the throughput test")
    args = parser.parse_args()

    if not os.path.exists(args.scene):
        raise SystemExit(
            f"scene not found: {args.scene}\nFetch it with the sparse-checkout in this "
            "file's docstring (into the gitignored third_party/)."
        )

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    summarize(model, data)

    # --- reset to the home pose ------------------------------------------------
    go_home(model, data)
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    home_hand = data.xpos[hand].copy()
    print("\n=== home pose ===")
    print(f"  qpos[:7] (arm)   : {np.round(data.qpos[:7], 3)}")
    print(f"  hand xyz         : {np.round(home_hand, 4)}")

    # --- render (free camera; the scene defines no named camera) ---------------
    print("\n=== render ===")
    try:
        with mujoco.Renderer(model, height=RENDER_HW, width=RENDER_HW) as r:
            r.update_scene(data)
            img = r.render()
        print(f"  frame {img.shape} {img.dtype}  std={img.std():.1f}  (non-blank: {img.std() > 1})")
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        try:
            import imageio.v2 as imageio

            imageio.imwrite(args.save, img)
            print(f"  saved {args.save}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not save PNG: {exc})")
        render_ok = True
    except Exception as exc:  # noqa: BLE001
        print(f"  render unavailable (no GL context?): {exc}")
        render_ok = False

    # --- actuation: nudge joint1 and confirm the hand moves --------------------
    print("\n=== actuation ===")
    go_home(model, data)
    data.ctrl[0] = 0.6  # command joint1 to 0.6 rad
    for _ in range(500):
        mujoco.mj_step(model, data)
    moved_hand = data.xpos[hand].copy()
    disp = float(np.linalg.norm(moved_hand - home_hand))
    print(f"  commanded joint1 -> 0.6 rad; hand moved {disp * 100:.1f} cm "
          f"(joint1 now {data.qpos[0]:.3f} rad)")

    # --- timing: physics throughput -------------------------------------------
    print("\n=== timing ===")
    go_home(model, data)
    for _ in range(50):  # warmup
        mujoco.mj_step(model, data)
    n = args.steps
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data)
    dt = time.perf_counter() - t0
    steps_per_s = n / dt
    realtime = (n * model.opt.timestep) / dt
    print(f"  physics: {n} steps in {dt:.3f} s -> {steps_per_s:,.0f} steps/s "
          f"({realtime:.1f}x real-time)")

    # --- timing: render throughput --------------------------------------------
    if render_ok:
        go_home(model, data)
        with mujoco.Renderer(model, height=RENDER_HW, width=RENDER_HW) as r:
            r.update_scene(data)
            r.render()  # warmup
            m = 30
            t0 = time.perf_counter()
            for _ in range(m):
                r.update_scene(data)
                r.render()
            dt = time.perf_counter() - t0
        print(f"  render : {m} frames at {RENDER_HW}x{RENDER_HW} in {dt:.3f} s "
              f"-> {m / dt:.1f} fps")

    # --- cadence mapping to the paper -----------------------------------------
    dt_sim = model.opt.timestep
    steps_per_action = round((1.0 / ACTION_FPS) / dt_sim)
    steps_per_clip = round((CLIP_FRAMES / ACTION_FPS) / dt_sim)
    print("\n=== cadence vs paper (4 fps, 0.25 s/action, 16-frame/4 s clip) ===")
    print(f"  0.25 s per action  = {steps_per_action} sim steps")
    print(f"  16-frame (4 s) clip = {steps_per_clip} sim steps")
    if render_ok:
        go_home(model, data)
        frames = []
        with mujoco.Renderer(model, height=RENDER_HW, width=RENDER_HW) as r:
            t0 = time.perf_counter()
            for _ in range(CLIP_FRAMES):
                for _ in range(steps_per_action):
                    mujoco.mj_step(model, data)
                r.update_scene(data)
                frames.append(r.render())
            clip_dt = time.perf_counter() - t0
        print(f"  built a {len(frames)}-frame clip (simulating 4 s of robot time) in "
              f"{clip_dt:.3f} s wall-clock")
        print("  => the sim side is cheap; the paper's ~16 s/action budget is dominated by "
              "the ViT-g CEM world model, not MuJoCo.")

    print("\nOK")


if __name__ == "__main__":
    main()
