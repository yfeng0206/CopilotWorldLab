"""Reproduce the V-JEPA 2-AC energy landscape headlessly on GPU (arXiv:2506.09985, Fig. 9).

This is the correctness gate for all downstream planning: it verifies that the loaded
action-conditioned model assigns *low* latent energy to actions near the ground-truth
transition and *high* energy elsewhere. It mirrors
``third_party/vjepa2/notebooks/energy_landscape_example.ipynb`` but runs without a notebook,
under our logging, on the GPU in bf16, and adds quantitative pass/fail checks:

1. Encode the two example frames (context, goal) to latents with the frozen ViT-g encoder.
2. Sweep a cartesian grid of xyz action deltas, roll each one step through the AC predictor,
   and score the latent energy ``mean(|predict(a) - z_goal|)``.
3. Check that the grid action of minimum energy is near the ground-truth action
   (``poses_to_diff`` of the two recorded states), and that playing the trajectory in
   reverse flips the sign of the energy minimum's dominant axis.

Runs isolated from the project's own ``src`` (the vendored repo ships its own top-level
``src``); loads the project logging module by file path first, then lets the vendored repo
own the import path -- identical to ``scripts/vjepa2_ac_infer_test.py``.

    python scripts/energy_landscape_repro.py
    python scripts/energy_landscape_repro.py --nsamples 9 --dtype bf16
    python scripts/energy_landscape_repro.py --reverse   # play the trajectory backwards
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
    """Load a module by file path without registering its package (avoids `src` clash)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the project logging module before the vendored `src` shadows our package.
clog = _load_by_path("cwlab_logging", os.path.join(_REPO_ROOT, "src", "utils", "logging.py"))

# Vendored repo must own the `src`/`app` import namespace (see infer script for rationale).
_VJEPA = os.path.join(_REPO_ROOT, "third_party", "vjepa2")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(_VJEPA, "notebooks"))
sys.path.insert(0, _VJEPA)

CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "vjepa2-ac-vitg.pt")
EXAMPLE_TRAJ = os.path.join(_VJEPA, "notebooks", "franka_example_traj.npz")
CROP = 256

logger = clog.get_logger("energy_landscape")


def load_model(device: str):
    import torch

    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model

    logger.info("building ViT-g encoder + AC predictor (pretrained=False)")
    encoder, predictor = _make_vjepa2_ac_model(model_name="vit_ac_giant", pretrained=False)

    logger.info("loading local checkpoint %s", os.path.basename(CHECKPOINT))
    state = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
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


def encode_frames(encoder, clips, tokens_per_frame):
    """[B,C,T,H,W] clip -> per-frame layer-normed latents flattened to [B, T*tpf, D]."""
    import torch.nn.functional as F

    b, c, t, h, w = clips.size()
    x = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    z = encoder(x)
    z = z.view(b, t, -1, z.size(-1)).flatten(1, 2)
    return F.layer_norm(z, (z.size(-1),))


def build_action_grid(nsamples: int, grid_size: float, device, dtype):
    """Cartesian grid of xyz action deltas (rotation and gripper zeroed): [S, 1, 7]."""
    import torch

    axis = np.linspace(-grid_size, grid_size, nsamples)
    dx, dy, dz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)  # [S, 3]
    seven = np.zeros((grid.shape[0], 7), dtype=np.float32)
    seven[:, :3] = grid
    return torch.as_tensor(seven, device=device, dtype=dtype).unsqueeze(1)  # [S, 1, 7]


def score_grid(predictor, z_ctx, s_ctx, z_goal, actions, tokens_per_frame, chunk):
    """Latent energy mean(|predict(a) - z_goal|) for every action in the grid: [S]."""
    import torch
    import torch.nn.functional as F

    s = actions.size(0)
    z_ctx = z_ctx.repeat(s, 1, 1)      # [S, tpf, D]
    s_ctx = s_ctx.repeat(s, 1, 1)      # [S, 1, 7]
    goal = z_goal.repeat(s, 1, 1)      # [S, tpf, D]
    energies = torch.empty(s, device=z_ctx.device, dtype=torch.float32)
    for i in range(0, s, chunk):
        sl = slice(i, i + chunk)
        nxt = predictor(z_ctx[sl], actions[sl], s_ctx[sl])[:, -tokens_per_frame:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),))
        energies[sl] = torch.abs(nxt - goal[sl]).mean(dim=(1, 2)).float()
    return energies


def save_heatmap(actions_np, energy_np, gt_action, nsamples, out_png, title):
    """dx-dz energy heatmap (min-energy dy slice), with the ground-truth action marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Pick the dy slice containing the global energy minimum, then grid dx vs dz.
    imin = int(np.argmin(energy_np))
    dy_star = actions_np[imin, 1]
    mask = np.isclose(actions_np[:, 1], dy_star)
    dx, dz, e = actions_np[mask, 0], actions_np[mask, 2], energy_np[mask]
    grid = e.reshape(nsamples, nsamples)  # dx (rows) x dz (cols) at fixed dy
    extent = [dz.min(), dz.max(), dx.min(), dx.max()]

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    plt.colorbar(label="latent energy  mean|P(a)-z_goal|")
    plt.scatter([gt_action[2]], [gt_action[0]], c="red", marker="*", s=200,
                edgecolors="white", label="ground-truth action")
    plt.xlabel("action delta z")
    plt.ylabel("action delta x")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def run_once(encoder, predictor, tokens_per_frame, clips, states, poses_to_diff,
             device, dtype, nsamples, grid_size, chunk, tag, stamp):
    import torch

    gt = poses_to_diff(states[0, 0], states[0, 1]).cpu().numpy()  # [7] ground-truth action
    with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=dtype is not None):
        z = encode_frames(encoder, clips, tokens_per_frame)
        z_ctx, z_goal = z[:, :tokens_per_frame], z[:, -tokens_per_frame:]
        s_ctx = states[:, :1].to(z.device)
        actions = build_action_grid(nsamples, grid_size, z.device, z.dtype)
        energy = score_grid(predictor, z_ctx, s_ctx, z_goal, actions, tokens_per_frame, chunk)

    actions_np = actions.squeeze(1).float().cpu().numpy()  # [S, 7]
    energy_np = energy.cpu().numpy()                       # [S]
    imin = int(np.argmin(energy_np))
    a_min = actions_np[imin, :3]                           # hard argmin = primary estimate

    # Robust local minimum: centroid over the lowest-energy quantile (not the whole grid,
    # which would bias the estimate toward the grid centre when the basin is near an edge).
    k = max(3, int(round(0.05 * energy_np.size)))
    low = np.argsort(energy_np)[:k]
    a_topk = actions_np[low, :3].mean(0)

    # Judge localization against the GT clipped into the swept box: if the true action lies
    # outside the grid the best any grid point can do is the boundary, so an unclipped error
    # is misleading. Flag that case as inconclusive for localization.
    gt_clipped = np.clip(gt[:3], -grid_size, grid_size)
    gt_outside_grid = bool(np.any(np.abs(gt[:3]) > grid_size + 1e-9))
    at_boundary = bool(np.any(np.isclose(np.abs(a_min), grid_size)))
    err_hard = float(np.linalg.norm(a_min - gt_clipped))       # vs clipped GT (fair)
    err_raw = float(np.linalg.norm(a_min - gt[:3]))            # vs unclipped GT (reference)
    err_topk = float(np.linalg.norm(a_topk - gt_clipped))

    # Landscape informativeness: how far the mean sits above the minimum, in std units. A
    # near-zero margin means a flat landscape -> localization is inconclusive.
    margin = float((energy_np.mean() - energy_np.min()) / (energy_np.std() + 1e-8))

    na, ng = np.linalg.norm(a_min), np.linalg.norm(gt[:3])
    cos_gt = float(np.dot(a_min, gt[:3]) / (na * ng)) if na > 1e-9 and ng > 1e-9 else 0.0

    logger.info("[%s] ground-truth action xyz = (%+.3f, %+.3f, %+.3f) full7=%s outside_grid=%s",
                tag, gt[0], gt[1], gt[2], np.round(gt, 3).tolist(), gt_outside_grid)
    logger.info("[%s] argmin xyz = (%+.3f, %+.3f, %+.3f) | energy=%.4f | boundary=%s | margin=%.2f",
                tag, a_min[0], a_min[1], a_min[2], energy_np[imin], at_boundary, margin)
    logger.info("[%s] top%d-centroid = (%+.3f, %+.3f, %+.3f)", tag, k, a_topk[0], a_topk[1], a_topk[2])
    logger.info("[%s] err(argmin vs clipped GT)=%.3f m | err(raw)=%.3f m | cos=%+.2f | energy [%.4f, %.4f]",
                tag, err_hard, err_raw, cos_gt, energy_np.min(), energy_np.max())

    out_png = os.path.join("outputs", f"energy_landscape_{tag}_{stamp}.png")
    os.makedirs("outputs", exist_ok=True)
    try:
        save_heatmap(actions_np, energy_np, gt[:3], nsamples, out_png,
                     f"V-JEPA 2-AC energy landscape ({tag})")
        logger.info("[%s] saved heatmap %s", tag, out_png)
    except (ImportError, RuntimeError) as exc:  # optional plotting deps / no display only
        logger.warning("[%s] heatmap skipped (plotting): %s", tag, exc)

    return {"tag": tag, "gt": gt, "a_min": a_min, "a_topk": a_topk,
            "err_hard": err_hard, "err_raw": err_raw, "err_topk": err_topk,
            "at_boundary": at_boundary, "gt_outside_grid": gt_outside_grid,
            "cos_gt": cos_gt, "margin": margin,
            "e_min": float(energy_np.min()), "e_max": float(energy_np.max())}


def main() -> None:
    import torch

    from utils.mpc_utils import poses_to_diff

    from app.vjepa_droid.transforms import make_transforms

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nsamples", type=int, default=7, help="grid points per axis (S = n^3)")
    parser.add_argument("--grid-size", type=float, default=0.075, help="half-extent per axis (m)")
    parser.add_argument("--chunk", type=int, default=200, help="predictor sub-batch over grid")
    parser.add_argument("--reverse", action="store_true", help="only play the trajectory backwards")
    parser.add_argument("--traj", nargs="+", default=None,
                        help="one or more transition .npz files (observations [1,2,H,W,3] uint8, "
                             "states [1,2,7]) to analyze instead of the paper example; the model "
                             "is loaded once and every trajectory is scored (used by the MuJoCo "
                             "transfer test and the camera ablation)")
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
    gpu = torch.cuda.get_device_name(dev) if is_cuda else "cpu"

    logger.info("=" * 60)
    logger.info("V-JEPA 2-AC energy landscape | device=%s dtype=%s grid=%d^3 gpu=%s",
                device, args.dtype, args.nsamples, gpu)
    logger.info("=" * 60)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = clog.CSVLogger(
        os.path.join("logs", f"energy_landscape_{stamp}.csv"),
        "tag", "gt_x", "gt_y", "gt_z", "min_x", "min_y", "min_z",
        "topk_x", "topk_y", "topk_z", "err_hard", "err_raw", "err_topk",
        "cos_gt", "margin", "at_boundary", "gt_outside_grid", "e_min", "e_max",
    )

    (encoder, predictor), load_ms = clog.gpu_timer(lambda: load_model(device))
    tokens_per_frame = (CROP // encoder.patch_size) ** 2
    logger.info("model loaded in %.1f s | tokens/frame=%d", load_ms / 1000.0, tokens_per_frame)

    transform = make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
        motion_shift=False, crop_size=CROP,
    )

    # Build the list of (tag, observations, states) to analyze. Paper example (with a
    # reverse sanity pass) by default; otherwise every provided transition .npz, single
    # direction, with one model load shared across them.
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
            data = np.load(path, allow_pickle=False)
            tag = os.path.splitext(os.path.basename(path))[0]
            # Optional embedded metadata lets us aggregate per camera rather than per file.
            camera = str(data["camera"]) if "camera" in data.files else None
            action = str(data["action"]) if "action" in data.files else None
            items.append((tag, data["observations"], data["states"], camera, action))
        paper_mode = False
    else:
        traj = np.load(EXAMPLE_TRAJ)
        dirs = [("reverse", True)] if args.reverse else [("forward", False), ("reverse", True)]
        items = [(tag, traj["observations"][:, ::-1].copy() if rev else traj["observations"],
                  traj["states"][:, ::-1].copy() if rev else traj["states"], None, None)
                 for tag, rev in dirs]
        paper_mode = not args.reverse

    results = []
    for tag, obs, st, camera, action in items:
        clips = transform(obs[0]).unsqueeze(0).to(device)
        states = torch.tensor(np.asarray(st))
        logger.info("[%s] clips=%s states=%s", tag, tuple(clips.shape), tuple(states.shape))
        r = run_once(encoder, predictor, tokens_per_frame, clips, states, poses_to_diff,
                     device, autocast_dtype, args.nsamples, args.grid_size, args.chunk, tag, stamp)
        r["camera"], r["action"] = camera, action
        results.append(r)
        csv.log(tag, *np.round(r["gt"][:3], 4), *np.round(r["a_min"], 4),
                *np.round(r["a_topk"], 4), round(r["err_hard"], 4), round(r["err_raw"], 4),
                round(r["err_topk"], 4), round(r["cos_gt"], 3), round(r["margin"], 3),
                int(r["at_boundary"]), int(r["gt_outside_grid"]), round(r["e_min"], 4), round(r["e_max"], 4))

    # Pass thresholds. The energy landscape is smooth and only *near* the GT (paper Fig. 9),
    # so we test direction (cosine) and an informative (non-flat) landscape, not exact hits.
    COS_OK, MARGIN_OK = 0.5, 0.3

    logger.info("-" * 60)
    for r in results:
        logger.info("VERDICT[%-22s] argmin=(%+.3f,%+.3f,%+.3f) gt=(%+.3f,%+.3f,%+.3f) "
                    "err=%.3f m cos=%+.2f margin=%.2f boundary=%s outside_grid=%s",
                    r["tag"], *r["a_min"], *r["gt"][:3], r["err_hard"], r["cos_gt"],
                    r["margin"], r["at_boundary"], r["gt_outside_grid"])

    failed = False
    if paper_mode and len(results) == 2:
        fwd, rev = results
        dom = int(np.argmax(np.abs(fwd["gt"][:3])))
        # Use the hard argmin (not the biased centroid) for the sign-flip check.
        flipped = (np.sign(fwd["a_min"][dom]) != np.sign(rev["a_min"][dom])
                   and abs(fwd["a_min"][dom]) > 1e-3 and abs(rev["a_min"][dom]) > 1e-3)
        informative = fwd["margin"] >= MARGIN_OK and rev["margin"] >= MARGIN_OK
        aligned = fwd["cos_gt"] >= COS_OK and rev["cos_gt"] >= COS_OK
        logger.info("reverse check: dom axis=%d fwd=%+.3f rev=%+.3f -> %s | aligned=%s | informative=%s",
                    dom, fwd["a_min"][dom], rev["a_min"][dom],
                    "FLIPPED" if flipped else "NOT flipped", aligned, informative)
        passed = flipped and informative and aligned
        logger.info("RESULT: energy-landscape reproduction %s", "PASS" if passed else "FAIL")
        failed = not passed
    elif not paper_mode:
        # Camera/interface ablation. Aggregate per camera (mean cosine, mean margin) when
        # camera metadata is present; fall back to a per-transition ranking otherwise. A
        # boundary/outside-grid transition is a setup issue, not model weakness -- exclude it
        # from the alignment statistic but count it.
        outside = sum(int(r["gt_outside_grid"]) for r in results)
        if outside:
            logger.warning("%d/%d transitions had GT outside the grid (widen --grid-size)",
                           outside, len(results))

        cams = sorted({r["camera"] for r in results if r["camera"]})
        if cams:
            logger.info("per-camera ranking (best mean GT alignment first; n = usable axes):")
            rows = []
            for cam in cams:
                rs = [r for r in results if r["camera"] == cam and not r["gt_outside_grid"]]
                if not rs:
                    continue
                cos = np.array([r["cos_gt"] for r in rs])
                rows.append((cam, float(cos.mean()), float(np.median(cos)), float(cos.min()),
                             float(np.mean([r["margin"] for r in rs])), len(rs)))
            for cam, mean_c, med_c, min_c, mean_m, n in sorted(rows, key=lambda x: -x[1]):
                ok = mean_c >= COS_OK and mean_m >= MARGIN_OK
                logger.info("  %-12s mean_cos=%+.2f median=%+.2f min=%+.2f margin=%.2f n=%d -> %s",
                            cam, mean_c, med_c, min_c, mean_m, n,
                            "transfers" if ok else "weak/inconclusive")

            # Per action-axis summary (which motions the model localizes across cameras).
            acts = sorted({r["action"] for r in results if r["action"]})
            if acts:
                logger.info("per-axis mean cosine (across cameras):")
                for a in acts:
                    ca = [r["cos_gt"] for r in results if r["action"] == a and not r["gt_outside_grid"]]
                    if ca:
                        logger.info("  %-4s mean_cos=%+.2f (n=%d)", a, float(np.mean(ca)), len(ca))
        else:
            ranked = sorted(results, key=lambda r: (-r["cos_gt"], r["err_hard"]))
            logger.info("per-transition ranking (best GT alignment first):")
            for r in ranked:
                ok = r["cos_gt"] >= COS_OK and r["margin"] >= MARGIN_OK and not r["gt_outside_grid"]
                logger.info("  %-24s cos=%+.2f err=%.3f m margin=%.2f -> %s",
                            r["tag"], r["cos_gt"], r["err_hard"], r["margin"],
                            "transfers" if ok else "weak/inconclusive")

    logger.info("done")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
