"""Benchmark 1 -- V-JEPA 2-AC transition-prediction / action-ranking (arXiv:2506.09985).

The honest first benchmark from the project's evaluation plan (docs/experiments/benchmark_plan.md):
does the vanilla model *understand robot transitions*? For each transition (context frame,
goal frame, executed action) we score the latent energy E(a) = mean(|P(a; z, s) - z_goal|) of
the TRUE action against K random negative actions of the same magnitude, and report:

- rank_frac  : within-transition fraction of negatives with higher energy than the true action
               (this is the primary metric; chance 0.5, 1.0 = true beats all its own negatives)
- top1_acc   : fraction of transitions where the true action beats ALL its negatives
- gap_z      : (mean negative energy - true energy) / std, an effect size
- null rank  : the SAME test scored against a MISMATCHED goal (shuffled across transitions).
               If the model is image-goal-conditioned, rank_frac >> null_rank ~ 0.5.
- auroc_pool : pooled true(1)-vs-negative(0) separability by -energy. Labelled "pooled" because
               it mixes energies across transitions (partly a global-calibration signal), so
               rank_frac is the cleaner within-transition number.

This needs no custom environment or dataset: it runs on the paper's DROID example trajectory
and on our rendered MuJoCo transitions (--traj). It is the baseline the fine-tuned predictor
will be measured against. Runs isolated from the project's own `src` (the vendored repo ships
a colliding `src`).

    python scripts/benchmark_transition_scoring.py                          # DROID example (fwd+rev)
    python scripts/benchmark_transition_scoring.py --traj "outputs/transitions/t_az45_el45_*.npz"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_by_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clog = _load_by_path("cwlab_logging", os.path.join(_REPO_ROOT, "src", "utils", "logging.py"))

_VJEPA = os.path.join(_REPO_ROOT, "third_party", "vjepa2")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(_VJEPA, "notebooks"))
sys.path.insert(0, _VJEPA)

CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "vjepa2-ac-vitg.pt")
EXAMPLE_TRAJ = os.path.join(_VJEPA, "notebooks", "franka_example_traj.npz")
CROP = 256
MIN_MOTION = 0.02  # metres: skip transitions whose true translation is smaller (degenerate)

logger = clog.get_logger("benchmark_transition")


def load_model(device: str):
    import torch

    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model

    logger.info("building ViT-g encoder + AC predictor (pretrained=False)")
    encoder, predictor = _make_vjepa2_ac_model(model_name="vit_ac_giant", pretrained=False)
    logger.info("loading local checkpoint %s", os.path.basename(CHECKPOINT))
    state = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    missing_e, unexpected_e = encoder.load_state_dict(_clean_backbone_key(state["encoder"]), strict=False)
    stray = [k for k in missing_e if "rope" not in k.lower() and "freq" not in k.lower()]
    if unexpected_e or stray:
        raise RuntimeError(f"encoder checkpoint mismatch: {stray[:5]} / {unexpected_e[:5]}")
    predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=True)
    logger.info("encoder loaded (missing=%d unexpected=%d), predictor strict OK",
                len(missing_e), len(unexpected_e))
    return encoder.to(device).eval(), predictor.to(device).eval()


def encode_frames(encoder, clips, tokens_per_frame):
    import torch.nn.functional as F

    b, c, t, h, w = clips.size()
    x = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    z = encoder(x)
    z = z.view(b, t, -1, z.size(-1)).flatten(1, 2)
    return F.layer_norm(z, (z.size(-1),))


def score_actions(predictor, z_ctx, s_ctx, z_goal, actions, tokens_per_frame, chunk):
    """Latent energy mean(|predict(a) - z_goal|) for each action [S,1,7] -> [S]."""
    import torch
    import torch.nn.functional as F

    s = actions.size(0)
    z_ctx = z_ctx.repeat(s, 1, 1)
    s_ctx = s_ctx.repeat(s, 1, 1)
    goal = z_goal.repeat(s, 1, 1)
    energy = torch.empty(s, device=z_ctx.device, dtype=torch.float32)
    for i in range(0, s, chunk):
        sl = slice(i, i + chunk)
        nxt = predictor(z_ctx[sl], actions[sl], s_ctx[sl])[:, -tokens_per_frame:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),))
        energy[sl] = torch.abs(nxt - goal[sl]).mean(dim=(1, 2)).float()
    return energy


def random_negative_directions(true_xyz, k, rng):
    """K random xyz directions with the same norm as the true action (direction test)."""
    norm = float(np.linalg.norm(true_xyz))
    dirs = rng.standard_normal((k, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    return dirs * norm


def auroc(pos_scores, neg_scores):
    """AUROC = P(score_pos > score_neg) via a tie-aware rank statistic (no sklearn)."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + 1 + csum) / 2.0  # average rank per unique value
    ranks = avg_rank[inv]
    r_pos = ranks[:pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def rank_stats(energy):
    """Within-transition rank stats: fraction of negatives beaten, top1, effect size, energies."""
    e_true, e_neg = energy[0], energy[1:]
    return (float(np.mean(e_neg > e_true)), bool(np.all(e_neg > e_true)),
            float((e_neg.mean() - e_true) / (e_neg.std() + 1e-8)), float(e_true), e_neg)


def main() -> None:
    import torch

    from utils.mpc_utils import poses_to_diff

    from app.vjepa_droid.transforms import make_transforms

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--negatives", type=int, default=32, help="random negative actions per transition")
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj", nargs="+", default=None,
                        help="transition .npz glob(s); default = the paper DROID example (fwd+rev)")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        logger.error("missing checkpoint: %s", CHECKPOINT)
        raise SystemExit(1)

    requested = args.device if torch.cuda.is_available() else "cpu"
    dev = torch.device(requested)
    is_cuda = dev.type == "cuda"
    if is_cuda and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    if is_cuda:
        torch.cuda.set_device(dev)
    device = requested
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else None
    rng = np.random.default_rng(args.seed)

    logger.info("=" * 60)
    logger.info("V-JEPA 2-AC transition-scoring benchmark | device=%s dtype=%s K=%d",
                device, args.dtype, args.negatives)
    logger.info("=" * 60)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = clog.CSVLogger(
        os.path.join("logs", f"benchmark_transition_{stamp}.csv"),
        "tag", "camera", "e_true", "e_neg_mean", "rank_frac", "null_rank_frac", "top1", "gap_z",
    )

    (encoder, predictor), _ = clog.gpu_timer(lambda: load_model(device))
    tokens_per_frame = (CROP // encoder.patch_size) ** 2
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                                random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                                motion_shift=False, crop_size=CROP)

    if args.traj:
        import glob

        paths = []
        for pattern in args.traj:
            paths.extend(sorted(glob.glob(pattern)))
        if not paths:
            logger.error("no trajectories matched: %s", args.traj)
            raise SystemExit(1)
        items = []
        for path in paths:
            data = np.load(path)
            tag = os.path.splitext(os.path.basename(path))[0]
            camera = str(data["camera"]) if "camera" in data.files else "n/a"
            items.append((tag, camera, data["observations"], data["states"]))
    else:
        traj = np.load(EXAMPLE_TRAJ)
        items = [("droid_example_fwd", "droid", traj["observations"], traj["states"]),
                 ("droid_example_rev", "droid", traj["observations"][:, ::-1].copy(),
                  traj["states"][:, ::-1].copy())]

    # Pass 1: encode every transition and build its candidate action set (true + K negatives).
    trans = []
    for tag, camera, obs, st in items:
        clips = transform(obs[0]).unsqueeze(0).to(device)
        states = torch.tensor(np.asarray(st))
        gt = poses_to_diff(states[0, 0], states[0, 1]).cpu().numpy()
        true_xyz = gt[:3]
        if float(np.linalg.norm(true_xyz)) < MIN_MOTION:
            logger.warning("skip %s: near-zero true motion (%.4f m)", tag, float(np.linalg.norm(true_xyz)))
            continue
        cand = np.zeros((1 + args.negatives, 7), dtype=np.float32)
        cand[0, :3] = true_xyz
        cand[1:, :3] = random_negative_directions(true_xyz, args.negatives, rng)
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            z = encode_frames(encoder, clips, tokens_per_frame)
        trans.append({
            "tag": tag, "camera": camera,
            "z_ctx": z[:, :tokens_per_frame], "z_goal": z[:, -tokens_per_frame:],
            "s_ctx": states[:, :1].to(z.device),
            "actions": torch.as_tensor(cand, device=z.device, dtype=z.dtype).unsqueeze(1),
        })

    n = len(trans)
    if n == 0:
        logger.error("no usable transitions")
        raise SystemExit(1)

    # Pass 2: score each transition against its OWN goal and against a MISMATCHED (shuffled) goal.
    results, pos_pool, neg_pool = [], [], []
    for i, t in enumerate(trans):
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            e_real = score_actions(predictor, t["z_ctx"], t["s_ctx"], t["z_goal"],
                                   t["actions"], tokens_per_frame, args.chunk).cpu().numpy()
            null_goal = trans[(i + 1) % n]["z_goal"] if n > 1 else t["z_goal"]
            e_null = score_actions(predictor, t["z_ctx"], t["s_ctx"], null_goal,
                                   t["actions"], tokens_per_frame, args.chunk).cpu().numpy()
        rf, top1, gap, e_true, e_neg = rank_stats(e_real)
        null_rf, _, _, _, _ = rank_stats(e_null)
        results.append({"tag": t["tag"], "camera": t["camera"], "rank_frac": rf,
                        "null_rank_frac": null_rf, "top1": top1, "gap_z": gap})
        pos_pool.append(-e_true)
        neg_pool.extend((-e_neg).tolist())
        logger.info("  %-24s cam=%-10s e_true=%.4f e_neg=%.4f rank=%.2f null=%.2f top1=%s gap_z=%+.2f",
                    t["tag"], t["camera"], e_true, float(e_neg.mean()), rf, null_rf, top1, gap)
        csv.log(t["tag"], t["camera"], round(e_true, 4), round(float(e_neg.mean()), 4),
                round(rf, 3), round(null_rf, 3), int(top1), round(gap, 3))

    rank = float(np.mean([r["rank_frac"] for r in results]))
    null = float(np.mean([r["null_rank_frac"] for r in results]))
    top1 = float(np.mean([r["top1"] for r in results]))
    gap = float(np.mean([r["gap_z"] for r in results]))
    au = auroc(pos_pool, neg_pool)
    cams = sorted({r["camera"] for r in results})
    multi_cam = len(cams) > 1

    logger.info("-" * 60)
    # Within-transition rank_frac is the primary, calibration-free metric. The pooled AUROC
    # mixes energies across transitions, so it also reflects global energy calibration.
    scope = "uncalibrated mixed-camera" if multi_cam else "single view"
    logger.info("BENCHMARK (%s, n=%d, K=%d): rank_frac=%.3f (null=%.3f) top1=%.3f gap_z=%+.2f "
                "auroc_pool=%.3f", scope, len(results), args.negatives, rank, null, top1, gap, au)
    logger.info("image-conditioning check: real rank %.3f vs shuffled-goal null %.3f "
                "(real >> ~0.5 null => the model uses the goal image)", rank, null)

    if multi_cam:
        logger.info("per-camera rank_frac (the primary MuJoCo result; aggregate above blends views):")
        for cam in cams:
            rs = [r["rank_frac"] for r in results if r["camera"] == cam]
            ns = [r["null_rank_frac"] for r in results if r["camera"] == cam]
            logger.info("  %-12s rank_frac=%.3f null=%.3f (n=%d)",
                        cam, float(np.mean(rs)), float(np.mean(ns)), len(rs))
    logger.info("done")


if __name__ == "__main__":
    main()
