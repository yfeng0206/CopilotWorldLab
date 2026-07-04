"""Extract real DROID transitions from the LeRobot droid_100 dataset for transition scoring.

Benchmark-2 data path (docs/experiments/benchmark_plan.md). The standard sim manipulation
suites (robosuite, ManiSkill) do not run on Windows (lessons_learned #11/#18) and robomimic
does not host pre-rendered image datasets (HF ships only low_dim + raw; images need robosuite
rendering). The Windows-runnable established alternative is DROID itself -- the real-robot
dataset V-JEPA 2-AC was trained/evaluated on (arXiv:2506.09985, arXiv:2403.12945) -- which is
the fairest "does the model understand real robot transitions?" test.

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
import os

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "lerobot/droid_100"
LOCAL_DIR = os.path.join(_REPO_ROOT, "data", "droid_100")
OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "droid_transitions")
FPS = 15


def download(camera_key: str) -> str:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HXET", "0")
    return snapshot_download(
        DATASET, repo_type="dataset", local_dir=LOCAL_DIR,
        allow_patterns=["meta/*", "meta/**", "data/**",
                        f"videos/{camera_key}/**"],
    )


def load_episode_meta(root: str) -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    rows: list[dict] = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def read_episode_states(root: str, chunk_index: int, file_index: int,
                        from_index: int, to_index: int) -> np.ndarray:
    import pyarrow.parquet as pq

    path = os.path.join(root, "data", f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.parquet")
    table = pq.read_table(path, columns=["observation.state", "episode_index", "frame_index"])
    state = table.column("observation.state").to_pylist()[from_index:to_index]
    return np.asarray(state, dtype=np.float32)


def decode_window(video_path: str, from_ts: float, to_ts: float, length: int) -> np.ndarray:
    """Decode the episode's frames (time window [from_ts, to_ts]) from an av1 mp4."""
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
    imgs = np.stack([f[1] for f in frames], axis=0) if frames else np.empty((0,))
    return imgs[:length]


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
    args = parser.parse_args()

    root = download(args.camera)
    episodes = load_episode_meta(root)
    episodes = episodes[: args.max_episodes]
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(OUT_DIR, "*.npz")):
        os.remove(f)

    rng = np.random.default_rng(args.seed)
    ck, fk = f"videos/{args.camera}/chunk_index", f"videos/{args.camera}/file_index"
    fromk, tok = f"videos/{args.camera}/from_timestamp", f"videos/{args.camera}/to_timestamp"

    total, kept = 0, 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        length = int(ep["length"])
        states = read_episode_states(root, int(ep["data/chunk_index"]), int(ep["data/file_index"]),
                                     int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
        vpath = os.path.join(root, "videos", args.camera,
                             f"chunk-{int(ep[ck]):03d}", f"file-{int(ep[fk]):03d}.mp4")
        if not os.path.exists(vpath):
            print(f"episode {ep_idx}: missing video {vpath}, skipping")
            continue
        frames = decode_window(vpath, float(ep[fromk]), float(ep[tok]), length)
        n = min(len(frames), len(states))
        if n < args.stride + 1:
            print(f"episode {ep_idx}: only {n} frames decoded, skipping")
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
        total += len(starts)
        print(f"episode {ep_idx:3d}: frames={n:3d} valid={len(valid):2d} "
              f"kept_total={kept}")

    print(f"\nwrote {kept} transitions to {OUT_DIR} "
          f"(from {len(episodes)} episodes, stride={args.stride}, min_motion={args.min_motion} m)")


if __name__ == "__main__":
    main()
