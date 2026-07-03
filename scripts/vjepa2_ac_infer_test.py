"""Load V-JEPA 2-AC (ViT-g) from the local checkpoint and time one planned action.

Reproduces the paper's planning setup (arXiv:2506.09985): encode a context frame and a
goal image to latents, then run Cross-Entropy-Method MPC that minimizes the latent energy
|| P(a; z, s) - z_g ||_1 over sampled action trajectories. The paper reports ~16 s/action
for 800 samples x 10 iterations x horizon 1 on a single RTX 4090; this times the same
configuration (and a few others) on the local RTX 3090, with full logging to logs/.

Runs isolated from the project's own `src`: the vendored facebookresearch/vjepa2 repo
ships its own top-level `src`, so this script loads the project logging module by file
path (avoiding the `src` collision) and then lets the vendored repo own the import path.

    python scripts/vjepa2_ac_infer_test.py
    python scripts/vjepa2_ac_infer_test.py --dtype bf16
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_by_path(module_name: str, file_path: str):
    """Load a module by file path without registering its package (avoids `src` clash)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the project logging module before the vendored `src` shadows our package.
clog = _load_by_path("cwlab_logging", os.path.join(_REPO_ROOT, "src", "utils", "logging.py"))

# Vendored repo must own the `src`/`app` import namespace. Drop the repo root (and cwd) from
# sys.path and evict any already-imported project `src*` modules so the vendored packages win
# unambiguously, even under `python -m ...`, pytest, or an IDE runner.
_VJEPA = os.path.join(_REPO_ROOT, "third_party", "vjepa2")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(_VJEPA, "notebooks"))
sys.path.insert(0, _VJEPA)

CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "vjepa2-ac-vitg.pt")
EXAMPLE_TRAJ = os.path.join(_VJEPA, "notebooks", "franka_example_traj.npz")
CROP = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

# (label, samples, cem_steps, rollout); the paper's reported setting is flagged PAPER.
# Graded small -> big sweep so per-sample scaling (and any super-linear cost) is visible.
CONFIGS = [
    ("warmup", 25, 2, 1),
    ("50s_10i_h1", 50, 10, 1),
    ("100s_10i_h1", 100, 10, 1),
    ("200s_10i_h1", 200, 10, 1),
    ("400s_10i_h1", 400, 10, 1),
    ("PAPER_800s_10i_h1", 800, 10, 1),
]

logger = clog.get_logger("vjepa2_ac_infer")


def preprocess(frames: np.ndarray):
    """[T, H, W, C] -> [C, T, H, W] float, ImageNet-normalized (256x256 input)."""
    import torch

    x = torch.as_tensor(np.asarray(frames), dtype=torch.float32).permute(3, 0, 1, 2)
    mean = torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.as_tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return (x - mean) / std


def load_model(device: str):
    import torch

    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model

    logger.info("building ViT-g encoder + AC predictor (pretrained=False)")
    encoder, predictor = _make_vjepa2_ac_model(model_name="vit_ac_giant", pretrained=False)

    logger.info("loading local checkpoint %s", os.path.basename(CHECKPOINT))
    state = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    # strict=False mirrors the upstream loader; RoPE has no persistent buffers in this
    # release, so a correct checkpoint loads clean. Still fail loudly on a real mismatch
    # (unexpected keys, or missing keys that are not RoPE frequencies) so a wrong checkpoint
    # cannot silently run with unloaded weights and produce meaningless actions/timings.
    missing_e, unexpected_e = encoder.load_state_dict(_clean_backbone_key(state["encoder"]), strict=False)
    stray_missing = [k for k in missing_e if "rope" not in k.lower() and "freq" not in k.lower()]
    if unexpected_e or stray_missing:
        raise RuntimeError(
            f"encoder checkpoint mismatch: {len(stray_missing)} unexpected-missing "
            f"{stray_missing[:5]}, {len(unexpected_e)} unexpected {unexpected_e[:5]}")
    predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=True)
    logger.info("encoder loaded (missing=%d unexpected=%d), predictor strict OK",
                len(missing_e), len(unexpected_e))
    return encoder.to(device).eval(), predictor.to(device).eval()


def encode_frames(encoder, clips, tokens_per_frame, normalize=True):
    import torch.nn.functional as F

    b, c, t, h, w = clips.size()
    x = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    z = encoder(x)
    z = z.view(b, t, -1, z.size(-1)).flatten(1, 2)
    return F.layer_norm(z, (z.size(-1),)) if normalize else z


def main() -> None:
    import torch
    import torch.nn.functional as F

    from utils.mpc_utils import cem, compute_new_pose

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=200,
                        help="predictor sub-batch size over CEM samples (0 = whole batch); "
                             "small values cap peak memory and avoid allocator thrash at high sample counts")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        logger.error("missing checkpoint: %s", CHECKPOINT)
        raise SystemExit(1)

    requested = args.device if torch.cuda.is_available() else "cpu"
    dev = torch.device(requested)
    is_cuda = dev.type == "cuda"
    if is_cuda:
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(dev)  # make no-arg cuda APIs and event timers target this device
    device = requested  # string form for .to(...)
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else None
    gpu_name = torch.cuda.get_device_name(dev) if is_cuda else "cpu"
    logger.info("=" * 60)
    logger.info("V-JEPA 2-AC inference timing | device=%s dtype=%s chunk=%d gpu=%s",
                device, args.dtype, args.chunk, gpu_name)
    logger.info("=" * 60)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_csv = clog.CSVLogger(
        os.path.join("logs", f"vjepa2_ac_timing_{stamp}.csv"),
        "config", "samples", "cem_steps", "rollout", "time_s", "predictor_s", "pose_s", "peak_gib",
    )

    _, load_ms = clog.gpu_timer(lambda: None)  # prime CUDA context
    (encoder, predictor), load_ms = clog.gpu_timer(lambda: load_model(device))
    tokens_per_frame = (CROP // encoder.patch_size) ** 2
    if is_cuda:
        weights_gib = torch.cuda.memory_allocated(dev) / 2**30
        logger.info("model loaded in %.1f s | weights on GPU %.2f GiB | tokens/frame=%d",
                    load_ms / 1000.0, weights_gib, tokens_per_frame)

    traj = np.load(EXAMPLE_TRAJ)
    frames = traj["observations"][0]
    states = torch.tensor(traj["states"])
    clips = preprocess(frames).unsqueeze(0).to(device)
    logger.info("trajectory frames=%s states=%s clips=%s",
                tuple(frames.shape), tuple(states.shape), tuple(clips.shape))

    # Per-config breakdown: predictor forward (GPU) vs pose update (CPU scipy) time.
    timers = {"predictor_s": 0.0, "pose_s": 0.0}

    def step_predictor(reps, actions, poses):
        b, t, n_t, d = reps.size()
        reps_flat = reps.flatten(1, 2)
        if is_cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        chunk = args.chunk if args.chunk > 0 else b
        outs = []
        for i in range(0, b, chunk):
            sl = slice(i, i + chunk)
            outs.append(predictor(reps_flat[sl], actions[sl], poses[sl])[:, -tokens_per_frame:])
        nxt = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(b, 1, n_t, d)
        if is_cuda:
            torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        new_pose = compute_new_pose(poses[:, -1:], actions[:, -1:])
        timers["predictor_s"] += t1 - t0
        timers["pose_s"] += time.perf_counter() - t1
        return nxt, new_pose

    def plan_once(samples, cem_steps, rollout):
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                              enabled=autocast_dtype is not None):
            z = encode_frames(encoder, clips, tokens_per_frame)
            z_ctx, z_goal = z[:, :tokens_per_frame], z[:, -tokens_per_frame:]
            s_ctx = states[:, :1].to(device)
            return cem(context_frame=z_ctx, context_pose=s_ctx, goal_frame=z_goal,
                       world_model=step_predictor, rollout=rollout, samples=samples,
                       cem_steps=cem_steps, topk=10, maxnorm=0.075,
                       momentum_mean=0.15, momentum_std=0.75,
                       momentum_mean_gripper=0.15, momentum_std_gripper=0.15)

    logger.info("timing one action per config (paper reference: ~16 s on RTX 4090):")
    paper_action = None
    for label, samples, steps, rollout in CONFIGS:
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        timers["predictor_s"] = 0.0
        timers["pose_s"] = 0.0
        try:
            action, elapsed_ms = clog.gpu_timer(lambda: plan_once(samples, steps, rollout))
        except RuntimeError as exc:
            logger.exception("config %s failed: %s", label, str(exc)[:80])
            if is_cuda:
                torch.cuda.empty_cache()
            continue
        peak = torch.cuda.max_memory_allocated(dev) / 2**30 if is_cuda else 0.0
        elapsed_s = elapsed_ms / 1000.0
        logger.info("  %-20s samples=%-4d iters=%-2d h=%d -> %6.2f s | predictor %5.2f s | pose %6.2f s | peak %.2f GiB",
                    label, samples, steps, rollout, elapsed_s,
                    timers["predictor_s"], timers["pose_s"], peak)
        timing_csv.log(label, samples, steps, rollout, round(elapsed_s, 3),
                       round(timers["predictor_s"], 3), round(timers["pose_s"], 3), round(peak, 3))
        if "PAPER" in label:
            paper_action = action[0, 0].cpu().numpy()

    if paper_action is not None:
        logger.info("PAPER-config planned action (dx,dy,dz,gripper) = (%+.3f, %+.3f, %+.3f, %+.3f)",
                    paper_action[0], paper_action[1], paper_action[2], paper_action[6])
    logger.info("done")


if __name__ == "__main__":
    main()
