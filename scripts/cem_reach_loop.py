"""Phase 1 -- closed-loop CEM planning to a goal image in FrankaDroidEnv (arXiv:2506.09985).

Wires V-JEPA 2-AC into the actual control loop: render an observation from the validated
planning camera, encode it and a GOAL IMAGE to latents, run Cross-Entropy-Method MPC to find
the 7-D end-effector delta that minimizes the latent energy toward the goal, apply it to the
real Franka arm, and repeat (receding horizon). Supports a SEQUENCE of goal images
(sub-goals): when the current sub-goal is reached the controller advances to the next one --
the structure a pick or stack task needs (approach-goal image, then placed/stacked-goal image).

Everything is logged per step (energy, distance to the goal pose, the planned action, the EE
pose, CEM time) to logs/ for later tables and graphs; observation frames and goal images are
saved to outputs/ for videos.

Namespace: the loop needs BOTH this project's `src` (FrankaDroidEnv) and the vendored repo's
colliding `src` (model + CEM). We import our env FIRST (binding the classes), then evict `src`
and hand the import path to the vendored repo; already-imported env classes keep working.

    python scripts/cem_reach_loop.py --task reach
    python scripts/cem_reach_loop.py --task chain --samples 200 --max-steps 25
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


# --- 1. Import OUR env first (bind the classes before the vendored `src` shadows the name). ---
sys.path.insert(0, _REPO_ROOT)
from src.envs.franka_build import PLANNING_CAMERA  # noqa: E402
from src.envs.franka_droid_env import FrankaDroidEnv  # noqa: E402

clog = _load_by_path("cwlab_logging", os.path.join(_REPO_ROOT, "src", "utils", "logging.py"))

# --- 2. Switch the import path to the vendored repo (its `src`/`app`/`utils` win from here). ---
_VJEPA = os.path.join(_REPO_ROOT, "third_party", "vjepa2")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(_VJEPA, "notebooks"))
sys.path.insert(0, _VJEPA)

CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "vjepa2-ac-vitg.pt")
CROP = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0

logger = clog.get_logger("cem_reach_loop")


def preprocess(frame_hwc):
    """[H,W,3] uint8 -> [1,3,1,H,W] float, ImageNet-normalized (single-frame clip)."""
    import torch

    x = torch.as_tensor(np.asarray(frame_hwc), dtype=torch.float32).permute(2, 0, 1)
    mean = torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]


def load_model(device):
    import torch

    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model

    logger.info("building ViT-g encoder + AC predictor")
    encoder, predictor = _make_vjepa2_ac_model(model_name="vit_ac_giant", pretrained=False)
    state = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = encoder.load_state_dict(_clean_backbone_key(state["encoder"]), strict=False)
    stray = [k for k in missing if "rope" not in k.lower() and "freq" not in k.lower()]
    if unexpected or stray:
        raise RuntimeError(f"encoder checkpoint mismatch: {stray[:5]} / {unexpected[:5]}")
    predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=True)
    logger.info("model loaded (missing=%d unexpected=%d)", len(missing), len(unexpected))
    return encoder.to(device).eval(), predictor.to(device).eval()


def encode(encoder, frame_hwc, device, tokens_per_frame):
    import torch
    import torch.nn.functional as F

    clip = preprocess(frame_hwc).to(device)  # [1,3,1,H,W]
    x = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    z = encoder(x)
    z = z.view(1, 1, -1, z.size(-1)).flatten(1, 2)
    return F.layer_norm(z, (z.size(-1),))[:, -tokens_per_frame:]


TASKS = {
    # task -> list of sub-goal EE deltas from the home pose (each a [dx,dy,dz,dR,dP,dY,dgrip]).
    "reach": [[0.10, 0.0, 0.0, 0, 0, 0, 0.0]],
    "chain": [[0.10, 0.0, 0.0, 0, 0, 0, 0.0], [0.10, 0.10, 0.0, 0, 0, 0, 0.0]],
}


def main() -> None:
    import torch
    import torch.nn.functional as F

    from utils.mpc_utils import cem, compute_new_pose

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=list(TASKS), default="reach")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=100, help="CEM population per action")
    parser.add_argument("--cem-steps", type=int, default=10)
    parser.add_argument("--rollout", type=int, default=1, help="CEM planning horizon")
    parser.add_argument("--max-steps", type=int, default=25, help="max env steps per sub-goal")
    parser.add_argument("--pos-tol", type=float, default=0.03, help="reached if EE within this of goal (m)")
    parser.add_argument("--maxnorm", type=float, default=0.075, help="CEM per-axis action clip (m)")
    parser.add_argument("--save-frames", action="store_true", help="save obs/goal frames to outputs/")
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

    logger.info("=" * 60)
    logger.info("Phase 1 CEM closed-loop | task=%s samples=%d cem_steps=%d rollout=%d dtype=%s",
                args.task, args.samples, args.cem_steps, args.rollout, args.dtype)
    logger.info("=" * 60)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_csv = clog.CSVLogger(
        os.path.join("logs", f"cem_loop_{args.task}_{stamp}.csv"),
        "subgoal", "step", "energy", "dist_to_goal_m", "dx", "dy", "dz", "dgrip",
        "rdx", "rdy", "rdz", "track_err_m", "ik_pos_err", "action_ok",
        "ee_x", "ee_y", "ee_z", "cem_time_s", "reached",
    )

    (encoder, predictor), load_ms = clog.gpu_timer(lambda: load_model(device))
    tokens_per_frame = (CROP // encoder.patch_size) ** 2
    logger.info("model loaded in %.1f s | tokens/frame=%d", load_ms / 1000.0, tokens_per_frame)

    env = FrankaDroidEnv(render_width=CROP, render_height=CROP, max_translation=0.13)
    env.reset()
    home_pos = env.get_ee_state()[:3].copy()

    # Build the goal-image sequence: teleport-preview the arm at each sub-goal pose and render.
    # Encode goals under the same no_grad/autocast path as the observations so their latents
    # are directly comparable (audit: z_goal and z_ctx must use the same dtype path).
    goals = []
    for delta in TASKS[args.task]:
        goal_pos = home_pos + np.asarray(delta[:3])
        goal_img = env.capture_goal_image(pos=goal_pos, euler=[np.pi, 0, 0], camera="planning")
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            z_goal = encode(encoder, goal_img, device, tokens_per_frame).detach()
        goals.append({"pos": goal_pos, "z_goal": z_goal, "img": goal_img})
    logger.info("built %d sub-goal image(s) for task '%s'", len(goals), args.task)
    env.reset()  # start the episode from home

    if args.save_frames:
        os.makedirs(os.path.join("outputs", f"cem_{args.task}_{stamp}"), exist_ok=True)
        try:
            import imageio.v2 as imageio

            for gi, g in enumerate(goals):
                imageio.imwrite(os.path.join("outputs", f"cem_{args.task}_{stamp}", f"goal_{gi}.png"), g["img"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("goal frame save skipped: %s", exc)

    def step_predictor(reps, actions, poses):
        b, t, n_t, d = reps.size()
        nxt = predictor(reps.flatten(1, 2), actions, poses)[:, -tokens_per_frame:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(b, 1, n_t, d)
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    def plan(z_ctx, s_ctx, z_goal):
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            # Freeze the gripper (axis 3 of the sampled xyz+gripper action) for reach tasks so
            # CEM does not spend samples on -- or the loop apply -- spurious gripper motion.
            return cem(context_frame=z_ctx, context_pose=s_ctx, goal_frame=z_goal,
                       world_model=step_predictor, rollout=args.rollout, samples=args.samples,
                       cem_steps=args.cem_steps, topk=10, maxnorm=args.maxnorm, axis={3: 0.0},
                       momentum_mean=0.15, momentum_std=0.75,
                       momentum_mean_gripper=0.15, momentum_std_gripper=0.15)

    import time

    total_reached = 0
    for gi, g in enumerate(goals):
        z_goal = g["z_goal"]
        logger.info("--- sub-goal %d/%d, target EE xyz=(%.3f,%.3f,%.3f) ---",
                    gi + 1, len(goals), *g["pos"])
        reached = False
        dist = float(np.linalg.norm(env.get_ee_state()[:3] - g["pos"]))
        for step in range(args.max_steps):
            state = env.get_ee_state()
            dist = float(np.linalg.norm(state[:3] - g["pos"]))
            if dist <= args.pos_tol:  # check BEFORE planning so post-action success is caught
                reached = True
                logger.info("  g%d step %2d | dist=%.3f m | REACHED (pre-plan)", gi, step, dist)
                break
            obs = env.render(camera="planning")
            with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                                 enabled=autocast_dtype is not None):
                z_ctx = encode(encoder, obs, device, tokens_per_frame)
                energy = float(F.l1_loss(z_ctx, z_goal).item())
            s_ctx = torch.tensor(state, dtype=torch.float32).view(1, 1, 7).to(device)
            t0 = time.perf_counter()
            action = plan(z_ctx, s_ctx, z_goal)[0, 0].float().cpu().numpy()
            action[6] = 0.0  # enforce gripper freeze for reach tasks
            cem_time = time.perf_counter() - t0

            # Apply, then measure the REALIZED delta vs the commanded action. The model was
            # trained on realized pose deltas, so a large commanded-vs-realized tracking error
            # (not the model) could cause near-goal drift -- log it to tell them apart.
            next_state = env.apply_action(action)
            realized = next_state[:3] - state[:3]
            track_err = float(np.linalg.norm(realized - action[:3]))
            step_csv.log(gi, step, round(energy, 5), round(dist, 4),
                         *[round(float(a), 4) for a in (action[0], action[1], action[2], action[6])],
                         *[round(float(r), 4) for r in realized], round(track_err, 4),
                         round(env.last_ik_pos_err, 4), int(env.last_action_ok),
                         *[round(float(p), 4) for p in state[:3]], round(cem_time, 3), int(reached))
            logger.info("  g%d step %2d | energy=%.4f | dist=%.3f | a=(%+.3f,%+.3f,%+.3f) "
                        "realized=(%+.3f,%+.3f,%+.3f) track_err=%.3f | %.1fs",
                        gi, step, energy, dist, action[0], action[1], action[2],
                        realized[0], realized[1], realized[2], track_err, cem_time)

        if not reached:
            dist = float(np.linalg.norm(env.get_ee_state()[:3] - g["pos"]))
            reached = dist <= args.pos_tol
        if reached:
            total_reached += 1
        else:
            logger.info("  g%d NOT reached within %d steps (final dist %.3f m)", gi, args.max_steps, dist)
            break  # do not chain past a missed sub-goal

    success = total_reached == len(goals)
    logger.info("-" * 60)
    logger.info("RESULT task=%s: %d/%d sub-goals reached -> %s",
                args.task, total_reached, len(goals), "SUCCESS" if success else "FAIL")
    logger.info("done")
    env.close()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
