"""Interactive MuJoCo viewer to replay recorded V-JEPA rollouts step by step.

Loads per-step qpos rollouts saved by ``run_closed_loop_benchmark.py --replay-record <dir>`` and
plays them back frozen (physics not stepped), so you see exactly the state the arm was in at every
control step -- the reach, the scripted close/lift, the final. Step with N (next) / B (back). Each
frame prints its condition, trial, phase, and scalars (dist-to-goal, latent energy, held, gripper
cmd, tilt) so you can see *where* a grasp misses.

Give one or more rollout dirs with labels, played in order. Typical use (frozen then planned):

    python scripts/replay_rollout_viewer.py --object cup \
        --dir logs/rollouts/frozen  --label FROZEN \
        --dir logs/rollouts/planned --label PLANNED

All rollouts in a session must be the same object (cup or box); rerun for the other object.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

KEY_NEXT = ord("N")      # N = next frame (SPACE is reserved by the viewer for pause)
KEY_PREV = ord("B")      # B = previous frame
KEY_RIGHT = 262          # right arrow (alternate next)
KEY_LEFT = 263           # left arrow (alternate previous)


class _Arg(argparse.Action):
    """Collect interleaved --dir / --label pairs preserving order."""
    def __call__(self, parser, ns, values, option_string=None):
        pairs = getattr(ns, "pairs", None) or []
        pairs.append((option_string, values))
        ns.pairs = pairs


def _load_frames(dirs_labels, obj):
    """Flatten every (condition, trial-file)'s per-step qpos into one ordered frame list."""
    frames = []
    for label, d in dirs_labels:
        d_abs = d if os.path.isabs(d) else os.path.join(_REPO_ROOT, d)
        files = sorted(glob.glob(os.path.join(d_abs, f"*_{obj}_*.npz")))
        for f in files:
            z = np.load(f, allow_pickle=True)
            if str(z["object_type"]) != obj:
                continue
            name = os.path.splitext(os.path.basename(f))[0]
            qpos, phase = z["qpos"], z["phase"]
            n = len(qpos)
            for k in range(n):
                frames.append({
                    "label": label, "name": name, "k": k, "n": n,
                    "qpos": qpos[k], "phase": str(phase[k]),
                    "dist": float(z["dist"][k]), "energy": float(z["energy"][k]),
                    "tilt": float(z["tilt"][k]), "held": float(z["held"][k]),
                    "grip": float(z["grip"][k]),
                })
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", choices=["cup", "box"], default="cup")
    p.add_argument("--dir", action=_Arg, help="a rollout dir (repeatable; pair with --label)")
    p.add_argument("--label", action=_Arg, help="label for the preceding --dir")
    args = p.parse_args()

    # Reconstruct ordered (label, dir) pairs from the interleaved --dir/--label stream.
    pairs, cur_dir = [], None
    for opt, val in getattr(args, "pairs", []):
        if opt == "--dir":
            cur_dir = val
        elif opt == "--label":
            pairs.append((val, cur_dir))
    if not pairs:
        raise SystemExit("give at least one --dir <path> --label <name>")

    frames = _load_frames(pairs, args.object)
    if not frames:
        raise SystemExit(f"no {args.object} rollouts found under {[d for _, d in pairs]}")

    import mujoco
    import mujoco.viewer
    from src.envs.franka_build import build_franka_robotiq

    model = build_franka_robotiq(add_object=True, add_zone=True, object_type=args.object,
                                 add_distractors=True)
    data = mujoco.MjData(model)
    # Thread-safety: the key_callback runs on the viewer's INPUT thread, so it must NOT touch
    # model/data (concurrent mj_forward from two threads is a native crash, 0xC0000005). It only
    # updates the target index; the MAIN loop applies qpos + mj_forward, serializing all state access.
    state = {"i": 0, "last": -1}

    def apply(i):
        fr = frames[i]
        data.qpos[:] = fr["qpos"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        held = "-" if fr["held"] < 0 else int(fr["held"])
        print(f"[{i + 1:3d}/{len(frames)}] {fr['label']:7s} {fr['name']}  "
              f"step {fr['k'] + 1}/{fr['n']}  {fr['phase']:16s}  "
              f"dist={fr['dist']:.3f}  E={fr['energy']:.3f}  held={held}  grip={fr['grip']:+.2f}  "
              f"tilt={fr['tilt']:.1f}deg", flush=True)

    def key_callback(keycode):
        if keycode in (KEY_NEXT, KEY_RIGHT):
            state["i"] = (state["i"] + 1) % len(frames)
        elif keycode in (KEY_PREV, KEY_LEFT):
            state["i"] = (state["i"] - 1) % len(frames)

    print(f"=== replay: object={args.object}, {len(frames)} frames across "
          f"{len(pairs)} condition(s): {', '.join(l for l, _ in pairs)} ===")
    print("N = next step, B = previous step (or right/left arrow), mouse = orbit/zoom, "
          "close window = quit")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if state["i"] != state["last"]:        # apply only on the MAIN thread (no race)
                apply(state["i"])
                state["last"] = state["i"]
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
