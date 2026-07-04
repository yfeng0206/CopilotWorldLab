"""Extract real DROID transitions from the LeRobot droid_100 dataset for transition scoring.

Benchmark 1 (world-model transition sanity) data path (docs/experiments/benchmark_plan.md).
This is a real-robot TRANSITION benchmark -- it tests whether the model scores the executed
action lower in latent energy than random alternatives -- NOT a grasp/place TASK benchmark: the
extracted transitions carry no task-success labels, so this does not replace robomimic
Lift/Can/Square for task completion. DROID is used because it is the real-robot dataset V-JEPA
2-AC was trained/evaluated on (arXiv:2506.09985, arXiv:2403.12945) and runs on Windows.

Note on robomimic: pre-rendered image HDF5 is not hosted (HF ships only low_dim + raw) and the
robosuite ENV RUNTIME does not step on this Windows setup, but robomimic raw states CAN be
re-rendered on Windows with direct MuJoCo + patched robosuite assets, so Lift/Can/Square remain
the grasp/place task sources when task-success labels are needed.

This reads `lerobot/droid_100` (100 real teleop episodes, LeRobot v3.0: parquet states/actions
+ av1 mp4 exterior camera) and emits one .npz per sampled transition in the exact format
`benchmark_transition_scoring.py` expects:
    observations : uint8 [1, 2, H, W, 3]  (context frame, goal frame)
    states       : float32 [1, 2, 7]      ([x, y, z, roll, pitch, yaw, gripper])
    camera       : str tag
The true action is recovered downstream from the state delta (poses_to_diff), so only the xyz
columns of observation.state need to be metric EE position -- which they are in DROID.

    python scripts/extract_droid_transitions.py --max-episodes 20 --per-episode 15
    python scripts/benchmark_transition_scoring.py --traj "outputs/droid_transitions/*.npz"
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "lerobot/droid_100"
LOCAL_DIR = os.path.join(_REPO_ROOT, "data", "droid_100")
OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "droid_transitions")
FPS = 15


def download(camera_key: str, revision: str) -> tuple[str, str]:
    from huggingface_hub import HfApi, snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HXET", "0")
    root = snapshot_download(
        DATASET, repo_type="dataset", revision=revision, local_dir=LOCAL_DIR,
        allow_patterns=["meta/*", "meta/**", "data/**", f"videos/{camera_key}/**"],
    )
    try:
        sha = HfApi().dataset_info(DATASET, revision=revision).sha
    except Exception:
        sha = revision
    return root, sha


def load_episode_meta(root: str) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    rows: list[dict] = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def read_episode_data(root: str, chunk_index: int, file_index: int,
                      from_index: int, to_index: int) -> dict:
    """Return the episode's per-frame state, timestamp, and frame_index (for alignment checks)."""
    import pyarrow.parquet as pq

    path = os.path.join(root, "data", f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.parquet")
    table = pq.read_table(path, columns=["observation.state", "timestamp", "frame_index"])
    sl = slice(from_index, to_index)
    return {
        "state": np.asarray(table.column("observation.state").to_pylist()[sl], dtype=np.float32),
        "timestamp": np.asarray(table.column("timestamp").to_pylist()[sl], dtype=np.float64).reshape(-1),
        "frame_index": np.asarray(table.column("frame_index").to_pylist()[sl], dtype=np.int64).reshape(-1),
    }


def decode_window(video_path: str, from_ts: float, to_ts: float,
                  length: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode the episode's frames (time window [from_ts, to_ts]) from an av1 mp4.

    Returns (frames [N,H,W,3] uint8, times [N] seconds relative to the video start).
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    seek_to = max(from_ts - 0.5, 0.0)
    container.seek(int(seek_to / stream.time_base), stream=stream, backward=True, any_frame=False)
    frames: list[tuple[float, np.ndarray]] = []
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        if t < from_ts - 1e-3:
            continue
        if t > to_ts + 1e-3:
            break
        frames.append((t, frame.to_ndarray(format="rgb24")))
    container.close()
    frames.sort(key=lambda x: x[0])
    if not frames:
        return np.empty((0,)), np.empty((0,))
    imgs = np.stack([f[1] for f in frames], axis=0)[:length]
    times = np.asarray([f[0] for f in frames], dtype=np.float64)[:length]
    return imgs, times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="observation.images.exterior_image_1_left",
                        help="LeRobot video key to use as the observation camera")
    parser.add_argument("--tag", default="droid_ext1", help="camera tag written into each npz")
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--per-episode", type=int, default=15,
                        help="max transitions sampled per episode")
    parser.add_argument("--stride", type=int, default=5,
                        help="frame gap H between context and goal (15 fps => H=5 ~ 0.33 s)")
    parser.add_argument("--min-motion", type=float, default=0.02,
                        help="metres: skip transitions with smaller xyz translation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default="main",
                        help="lerobot/droid_100 dataset revision to pin (branch, tag, or commit)")
    parser.add_argument("--align-tol", type=float, default=0.5,
                        help="max |decoded frame time - parquet timestamp| / frame_period tolerated")
    args = parser.parse_args()

    root, revision_sha = download(args.camera, args.revision)
    episodes = load_episode_meta(root)
    episodes = episodes[: args.max_episodes]
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(OUT_DIR, "*.npz")):
        os.remove(f)

    rng = np.random.default_rng(args.seed)
    ck, fk = f"videos/{args.camera}/chunk_index", f"videos/{args.camera}/file_index"
    fromk, tok = f"videos/{args.camera}/from_timestamp", f"videos/{args.camera}/to_timestamp"

    total, kept, per_episode = 0, 0, {}
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        length = int(ep["length"])
        data = read_episode_data(root, int(ep["data/chunk_index"]), int(ep["data/file_index"]),
                                 int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
        states, ts, fidx = data["state"], data["timestamp"], data["frame_index"]
        # frame_index must be contiguous 0..length-1, else the parquet slice is misaligned.
        if not (fidx.size == length and np.array_equal(fidx, np.arange(length))):
            print(f"episode {ep_idx}: non-contiguous frame_index, skipping")
            continue
        vpath = os.path.join(root, "videos", args.camera,
                             f"chunk-{int(ep[ck]):03d}", f"file-{int(ep[fk]):03d}.mp4")
        if not os.path.exists(vpath):
            print(f"episode {ep_idx}: missing video {vpath}, skipping")
            continue
        from_ts = float(ep[fromk])
        frames, ftimes = decode_window(vpath, from_ts, float(ep[tok]), length)
        n = min(len(frames), len(states))
        if n < args.stride + 1:
            print(f"episode {ep_idx}: only {n} frames decoded, skipping")
            continue
        # Validate frame/state time alignment: decoded frame time (relative to from_ts) must match
        # the parquet per-frame timestamp within a fraction of the frame period. This guards against
        # PyAV drops/dupes silently pairing the wrong image with a state/action.
        rel = ftimes[:n] - from_ts
        misalign = np.abs(rel - ts[:n]) * FPS
        if float(misalign.max()) > args.align_tol:
            print(f"episode {ep_idx}: frame/state misalignment {misalign.max():.2f} frames "
                  f"(>{args.align_tol}), skipping")
            continue

        starts = np.arange(0, n - args.stride)
        xyz = states[:, :3]
        motion = np.linalg.norm(xyz[starts + args.stride] - xyz[starts], axis=1)
        valid = starts[motion >= args.min_motion]
        rng.shuffle(valid)
        valid = valid[: args.per_episode]
        for i in sorted(valid.tolist()):
            j = i + args.stride
            obs = np.stack([frames[i], frames[j]], axis=0)[None].astype(np.uint8)  # [1,2,H,W,3]
            st = np.stack([states[i], states[j]], axis=0)[None].astype(np.float32)  # [1,2,7]
            out = os.path.join(OUT_DIR, f"droid_ep{ep_idx:03d}_f{i:04d}.npz")
            np.savez_compressed(out, observations=obs, states=st, camera=args.tag)
            kept += 1
        per_episode[ep_idx] = int(len(valid))
        total += len(starts)
        print(f"episode {ep_idx:3d}: frames={n:3d} align_ok(max={misalign.max():.2f}f) "
              f"valid={len(valid):2d} kept_total={kept}")

    manifest = {
        "dataset": DATASET, "revision": args.revision, "revision_sha": revision_sha,
        "camera": args.camera, "tag": args.tag, "fps": FPS,
        "max_episodes": args.max_episodes, "per_episode": args.per_episode,
        "stride": args.stride, "min_motion_m": args.min_motion, "seed": args.seed,
        "align_tol_frames": args.align_tol, "n_transitions": kept,
        "per_episode_counts": per_episode,
    }
    man_dir = os.path.join(_REPO_ROOT, "results", "benchmarks")
    os.makedirs(man_dir, exist_ok=True)
    with open(os.path.join(man_dir, "droid_extraction_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {kept} transitions to {OUT_DIR} "
          f"(from {len(episodes)} episodes, stride={args.stride}, min_motion={args.min_motion} m)")
    print(f"provenance: {DATASET}@{revision_sha[:12]} -> results/benchmarks/droid_extraction_manifest.json")


if __name__ == "__main__":
    main()
