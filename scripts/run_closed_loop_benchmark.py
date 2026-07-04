"""Tiny smoke benchmark runner: closed-loop V-JEPA 2-AC on Reach / Grasp-Lift / Place.

The Phase-1 task-success benchmark (docs/experiments/closed_loop_success_plan.md), at SMOKE scale.
For each task the model sees only an RGB image (the validated PLANNING_CAMERA), the 7-D EE state,
and a goal image; V-JEPA 2-AC plans the coarse motion with CEM MPC. Scripted primitives handle the
gripper (close/lift/open). Success is judged ONLY from hidden privileged MuJoCo truth
(src/bench/success.py) -- object pose, contacts, velocity, tilt -- never from the latent energy.

CEM config (defaults follow Meta's released wrapper; smaller population for the 3090 smoke):
    samples=200, cem_steps=10, rollout/horizon T=2, topk=10, maxnorm=0.05 m/axis, momentum 0.15.
Later ablations (not here): T=1 vs T=2, samples 200/400/800, maxnorm 0.05 vs 0.075.

Per trial we write a per-step CSV (logs/) and a GIF + contact sheet (outputs/closed_loop_benchmark/
<run_id>/) with projected markers (red=object, blue=EE, green=zone) and a stats panel, for visual
audit BEFORE any success-rate is claimed.

    python scripts/run_closed_loop_benchmark.py --tasks reach grasp_lift place --trials 1
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
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- 1. Import OUR src first (bind classes/functions before the vendored `src` shadows it). ---
sys.path.insert(0, _REPO_ROOT)
from src.envs.franka_build import (  # noqa: E402
    CUBE_HALF,
    CUBE_START,
    PLANNING_CAMERA,
    TABLE_TOP_Z,
)
from src.envs.franka_droid_env import FrankaDroidEnv  # noqa: E402
from src.bench.schema import SUCCESS_DEFAULTS  # noqa: E402
from src.bench.success import (  # noqa: E402
    grasp_lift_success,
    place_success,
    reach_success,
)

clog = _load_by_path("cwlab_logging", os.path.join(_REPO_ROOT, "src", "utils", "logging.py"))

# --- 2. Switch the import path to the vendored repo (its `src`/`app`/`utils` win from here). ---
_VJEPA = os.path.join(_REPO_ROOT, "third_party", "vjepa2")
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _REPO_ROOT]
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]
sys.path.insert(0, os.path.join(_VJEPA, "notebooks"))
sys.path.insert(0, _VJEPA)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CHECKPOINT = os.path.join(_REPO_ROOT, "checkpoints", "vjepa2-ac-vitg.pt")
CROP = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
EE_DOWN = [np.pi, 0.0, 0.0]  # gripper pointing down (extrinsic XYZ euler)

logger = clog.get_logger("closed_loop_benchmark")


# ----------------------------------------------------------------------------- model / encode
def preprocess(frame_hwc):
    import torch

    x = torch.as_tensor(np.asarray(frame_hwc), dtype=torch.float32).permute(2, 0, 1)
    mean = torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]


def load_model(device):
    import torch

    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model

    encoder, predictor = _make_vjepa2_ac_model(model_name="vit_ac_giant", pretrained=False)
    state = torch.load(CHECKPOINT, map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = encoder.load_state_dict(_clean_backbone_key(state["encoder"]), strict=False)
    stray = [k for k in missing if "rope" not in k.lower() and "freq" not in k.lower()]
    if unexpected or stray:
        raise RuntimeError(f"encoder checkpoint mismatch: {stray[:5]} / {unexpected[:5]}")
    predictor.load_state_dict(_clean_backbone_key(state["predictor"]), strict=True)
    return encoder.to(device).eval(), predictor.to(device).eval()


def encode(encoder, frame_hwc, device, tokens_per_frame):
    import torch
    import torch.nn.functional as F

    clip = preprocess(frame_hwc).to(device)
    x = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    z = encoder(x)
    z = z.view(1, 1, -1, z.size(-1)).flatten(1, 2)
    return F.layer_norm(z, (z.size(-1),))[:, -tokens_per_frame:]


# ----------------------------------------------------------------------------- camera projection
def _cam_basis(cam=PLANNING_CAMERA):
    az, el = np.radians(cam["azimuth"]), np.radians(cam["elevation"])
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    cam_pos = np.asarray(cam["lookat"], float) - cam["distance"] * forward
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return cam_pos, forward, right, up


def project(points, fovy, H=CROP, W=CROP, cam=PLANNING_CAMERA):
    """Project world points [N,3] into the PLANNING_CAMERA image -> [N,2] pixel (u,v).

    Validated: green(zone)/blue(EE)/red(object) markers land on their bodies. Depth-behind
    points return NaN.
    """
    cam_pos, forward, right, up = _cam_basis(cam)
    f = 0.5 * H / np.tan(0.5 * np.radians(fovy))
    out = []
    for p in np.atleast_2d(np.asarray(points, float)):
        d = p - cam_pos
        z = d @ forward
        if z <= 1e-6:
            out.append((np.nan, np.nan))
            continue
        out.append((W / 2 + f * (d @ right) / z, H / 2 - f * (d @ up) / z))
    return np.asarray(out)


# ----------------------------------------------------------------------------- visualization
def render_panel(img, obj_px, ee_px, zone_px, zone_r_px, stats):
    """One annotated frame: camera image with markers + a stats text panel -> RGB uint8 array."""
    fig = plt.figure(figsize=(8.4, 4.2), dpi=100)
    ax = fig.add_axes([0.01, 0.02, 0.47, 0.96])
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    if zone_px is not None and np.isfinite(zone_px).all():
        ax.add_patch(plt.Circle((zone_px[0], zone_px[1]), max(zone_r_px, 3.0),
                                 fill=False, color="lime", lw=2.0))
    if ee_px is not None and np.isfinite(ee_px).all():
        ax.plot(ee_px[0], ee_px[1], "o", color="blue", ms=9, mec="white", mew=1.2)
    if obj_px is not None and np.isfinite(obj_px).all():
        ax.plot(obj_px[0], obj_px[1], "o", color="red", ms=9, mec="white", mew=1.2)
    axt = fig.add_axes([0.50, 0.02, 0.49, 0.96])
    axt.axis("off")
    axt.text(0.0, 1.0, "\n".join(f"{k:<12} {v}" for k, v in stats.items()),
             va="top", ha="left", family="monospace", fontsize=8.5)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return arr


def save_contact_sheet(frames, path, cols=6):
    if not frames:
        return
    n = len(frames)
    idx = np.unique(np.linspace(0, n - 1, min(cols * 2, n)).astype(int))
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 1.5))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for k, i in enumerate(idx):
        axes[k].imshow(frames[i])
        axes[k].set_title(f"f{i}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- privileged snapshot
def snapshot(env, has_object):
    ee = env.get_ee_state()
    s = {"ee": ee[:3].copy(), "grip": float(ee[6])}
    if has_object:
        s.update(obj=env.object_position(), tilt=env.object_tilt(), speed=env.object_speed(),
                 held=env.gripper_holds_object(), released=env.object_released(),
                 zone=env.zone_center())
    return s


class TrialLogger:
    """Accumulates per-step CSV rows and annotated frames for one trial."""

    def __init__(self, csv, task, trial, fovy, has_object):
        self.csv, self.task, self.trial, self.fovy, self.has_object = csv, task, trial, fovy, has_object
        self.frames = []
        self.step = 0

    def record(self, env, phase, action, realized, energy, target, cem_time,
               dist_goal, success="", failure=""):
        img = env.render(camera="planning")
        s = snapshot(env, self.has_object)
        obj = s.get("obj")
        ee = s["ee"]
        zone = s.get("zone")
        zone3 = np.array([zone[0], zone[1], TABLE_TOP_Z + 0.001]) if zone is not None else None
        obj_px = project(obj, self.fovy)[0] if obj is not None else None
        ee_px = project(ee, self.fovy)[0]
        zone_px = zone_r_px = None
        if zone3 is not None:
            zp = project([zone3, zone3 + np.array([SUCCESS_DEFAULTS["place"]["zone_radius"], 0, 0])],
                         self.fovy)
            zone_px = zp[0]
            zone_r_px = float(np.linalg.norm(zp[1] - zp[0])) if np.isfinite(zp).all() else 4.0
        obj_dz = tilt = speed = float("nan")
        held = released = ""
        if self.has_object:
            obj_dz = float(obj[2] - (TABLE_TOP_Z + CUBE_HALF))
            tilt, speed = float(np.degrees(s["tilt"])), float(s["speed"])
            held, released = int(s["held"]), int(s["released"])
        a = action if action is not None else [np.nan] * 7
        r = realized if realized is not None else [np.nan] * 3
        tgt = target if target is not None else [np.nan] * 3
        self.csv.log(self.task, self.trial, self.step, phase,
                     _r(energy), *[_r(x) for x in (a[0], a[1], a[2], a[6])],
                     *[_r(x) for x in r[:3]], *[_r(x) for x in ee[:3]],
                     *([_r(x) for x in obj[:3]] if obj is not None else ["", "", ""]),
                     *[_r(x) for x in tgt[:3]], _r(dist_goal), _r(obj_dz), _r(tilt), _r(speed),
                     held, released, _r(cem_time), success, failure)
        stats = {
            "task": self.task, "trial": self.trial, "step": self.step, "phase": phase,
            "energy": _fmt(energy), "action": _vec(a[:3]), "realized": _vec(r[:3]),
            "grip_cmd": _fmt(a[6]), "ee_xyz": _vec(ee[:3]),
            "obj_xyz": _vec(obj[:3]) if obj is not None else "-",
            "target": _vec(tgt[:3]), "dist_goal": _fmt(dist_goal),
            "obj_dz": _fmt(obj_dz), "tilt_deg": _fmt(tilt), "obj_speed": _fmt(speed),
            "held": held, "released": released,
            "cem_time_s": _fmt(cem_time), "success": success, "failure": failure,
        }
        self.frames.append(render_panel(img, obj_px, ee_px, zone_px, zone_r_px or 4.0, stats))
        self.step += 1


def _r(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return ""


def _fmt(x):
    try:
        return f"{float(x):+.4f}" if np.isfinite(float(x)) else "-"
    except (TypeError, ValueError):
        return "-"


def _vec(v):
    return "(" + ",".join(f"{float(x):+.3f}" for x in v) + ")"


# ----------------------------------------------------------------------------- CEM planning
def make_planner(cem, compute_new_pose, predictor, tokens_per_frame, args, dev, autocast_dtype):
    import torch
    import torch.nn.functional as F

    def step_predictor(reps, actions, poses):
        b, t, n_t, d = reps.size()
        nxt = predictor(reps.flatten(1, 2), actions, poses)[:, -tokens_per_frame:]
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(b, 1, n_t, d)
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    def plan(z_ctx, s_ctx, z_goal):
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            return cem(context_frame=z_ctx, context_pose=s_ctx, goal_frame=z_goal,
                       world_model=step_predictor, rollout=args.rollout, samples=args.samples,
                       cem_steps=args.cem_steps, topk=args.topk, maxnorm=args.maxnorm,
                       axis={3: 0.0}, momentum_mean=0.15, momentum_std=0.75,
                       momentum_mean_gripper=0.15, momentum_std_gripper=0.15)

    return step_predictor, plan


def cem_to_goal(env, encoder, predictor_plan, z_goal, target, tlog, phase, max_steps,
                pos_tol, encode_fn, device, tokens_per_frame, dev, autocast_dtype):
    """Closed-loop CEM to a goal image; gripper frozen. Returns steps taken and final distance."""
    import torch
    import torch.nn.functional as F

    dist = float(np.linalg.norm(env.get_ee_state()[:3] - target))
    steps = 0
    for _ in range(max_steps):
        state = env.get_ee_state()
        dist = float(np.linalg.norm(state[:3] - target))
        if dist <= pos_tol:
            break
        obs = env.render(camera="planning")
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            z_ctx = encode_fn(encoder, obs, device, tokens_per_frame)
            energy = float(F.l1_loss(z_ctx, z_goal).item())
        s_ctx = torch.tensor(state, dtype=torch.float32).view(1, 1, 7).to(device)
        t0 = time.perf_counter()
        action = predictor_plan(z_ctx, s_ctx, z_goal)[0, 0].float().cpu().numpy()
        action[6] = 0.0  # gripper frozen during the V-JEPA reach
        cem_time = time.perf_counter() - t0
        nxt = env.apply_action(action)
        realized = nxt[:3] - state[:3]
        tlog.record(env, phase, action, realized, energy, target, cem_time, dist)
        steps += 1
    return steps, dist


# ----------------------------------------------------------------------------- scripted primitives
def scripted(env, tlog, phase, target, gripper=None, dz=None, settle=0, n=4):
    """A scripted EE move / gripper action, logged (no CEM). Returns nothing."""
    if target is not None:
        for _ in range(n):
            cur = env.get_ee_state()[:3]
            d = np.zeros(7)
            d[:3] = np.asarray(target) - cur
            env.apply_action(d)
    if gripper is not None:
        for _ in range(3):
            g = np.zeros(7)
            g[6] = gripper
            env.apply_action(g)
    if dz is not None:
        cur = env.get_ee_state()[:3]
        for _ in range(n):
            d = np.zeros(7)
            d[2] = dz / n
            env.apply_action(d)
    for _ in range(settle):
        env.apply_action(np.zeros(7))
    tlog.record(env, phase, None, None, float("nan"),
                target if target is not None else None, float("nan"), float("nan"))


# ----------------------------------------------------------------------------- tasks
def task_reach(env, ctx, tlog, args):
    env.reset()
    home = env.get_ee_state()[:3].copy()
    target = home + np.array([0.10, -0.08, -0.06])
    goal_img = env.capture_goal_image(pos=target, euler=EE_DOWN, camera="planning")
    z_goal = ctx["encode_goal"](goal_img)
    env.reset()
    steps, dist = cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, target, tlog,
                              "vjepa_reach", args.reach_steps, args.pos_tol, **ctx["cem_kw"])
    res = reach_success(env.get_ee_state()[:3], target, tau_reach=args.reach_tau)
    tlog.record(env, "final", None, None, float("nan"), target, float("nan"), dist,
                success=int(res.success), failure=res.failure_type or "")
    return res


def _grasp_goal(env):
    c = env.object_position()
    return np.array([c[0], c[1], c[2] + 0.005])


def task_grasp_lift(env, ctx, tlog, args):
    env.reset(cube_xy=CUBE_START[:2])
    grasp_pos = _grasp_goal(env)
    goal_img = env.capture_goal_image(pos=grasp_pos, euler=EE_DOWN, gripper=0.0, camera="planning")
    z_goal = ctx["encode_goal"](goal_img)
    env.reset(cube_xy=CUBE_START[:2])
    # 1) V-JEPA reach to the grasp-ready pose (gripper frozen open)
    cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, grasp_pos, tlog,
                "vjepa_reach", args.grasp_steps, args.pos_tol, **ctx["cem_kw"])
    # 2) scripted grasp primitive: align above the cube (vertical approach), descend, close,
    #    lift, settle. The align-above step keeps the descent vertical so the fingers straddle
    #    the cube instead of nudging it from the side (diagonal approach -> "pushed").
    c = env.object_position()
    scripted(env, tlog, "align", [c[0], c[1], c[2] + 0.12])
    c = env.object_position()
    scripted(env, tlog, "descend", [c[0], c[1], c[2] + 0.005])
    scripted(env, tlog, "close", None, gripper=1.0)
    obj_z0 = float(env.object_position()[2])
    scripted(env, tlog, "lift", None, dz=0.08)
    scripted(env, tlog, "settle", None, settle=20)
    obj = env.object_position()
    res = grasp_lift_success(obj_z0, float(obj[2]), env.get_ee_state()[:2], obj[:2],
                             env.object_tilt(), env.object_speed(), env.gripper_holds_object(),
                             SUCCESS_DEFAULTS["grasp_lift"])
    tlog.record(env, "final", None, None, float("nan"), None, float("nan"), float("nan"),
                success=int(res.success), failure=res.failure_type or "")
    return res


def task_place(env, ctx, tlog, args):
    env.reset(cube_xy=CUBE_START[:2])
    # scripted grasp + lift first so we start holding the cube reliably (place tests the release)
    c = env.object_position()
    scripted(env, tlog, "grasp_approach", [c[0], c[1], c[2] + 0.12])
    scripted(env, tlog, "descend", [c[0], c[1], c[2] + 0.005])
    scripted(env, tlog, "close", None, gripper=1.0)
    scripted(env, tlog, "lift", None, dz=0.10)
    if not env.gripper_holds_object():
        res = place_success(env.object_position()[:2], env.zone_center(), env.object_tilt(),
                            env.object_speed(), env.object_released(), SUCCESS_DEFAULTS["place"])
        tlog.record(env, "final", None, None, float("nan"), None, float("nan"), float("nan"),
                    success=int(res.success), failure="grasp_failed_pre_place")
        return res
    # place goal: EE over the zone at a low hover, then open
    zone = env.zone_center()
    place_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.12])
    goal_img = env.capture_goal_image(pos=place_pos, euler=EE_DOWN, gripper=1.0, camera="planning")
    z_goal = ctx["encode_goal"](goal_img)
    cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, place_pos, tlog,
                "vjepa_place", args.place_steps, args.pos_tol, **ctx["cem_kw"])
    # lower, open, settle
    zc = env.zone_center()
    scripted(env, tlog, "lower", [zc[0], zc[1], TABLE_TOP_Z + CUBE_HALF + 0.04])
    scripted(env, tlog, "open", None, gripper=-1.0)
    scripted(env, tlog, "settle", None, settle=30)
    obj = env.object_position()
    res = place_success(obj[:2], env.zone_center(), env.object_tilt(), env.object_speed(),
                        env.object_released(), SUCCESS_DEFAULTS["place"])
    tlog.record(env, "final", None, None, float("nan"), None, float("nan"), float("nan"),
                success=int(res.success), failure=res.failure_type or "")
    return res


TASKS = {"reach": task_reach, "grasp_lift": task_grasp_lift, "place": task_place}


# ----------------------------------------------------------------------------- main
def main() -> None:
    import torch

    from utils.mpc_utils import cem, compute_new_pose

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["reach", "grasp_lift", "place"], choices=list(TASKS))
    p.add_argument("--trials", type=int, default=1, help="trials per task (smoke: 1; real smoke: 3-5)")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--cem-steps", type=int, default=10)
    p.add_argument("--rollout", type=int, default=2, help="CEM planning horizon T")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--maxnorm", type=float, default=0.05, help="CEM per-axis action clip (m)")
    p.add_argument("--pos-tol", type=float, default=0.03, help="reached if EE within this of goal (m)")
    p.add_argument("--reach-tau", type=float, default=0.05, help="reach success threshold (m)")
    p.add_argument("--reach-steps", type=int, default=5)
    p.add_argument("--grasp-steps", type=int, default=6)
    p.add_argument("--place-steps", type=int, default=8)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not os.path.exists(CHECKPOINT):
        logger.error("missing checkpoint: %s", CHECKPOINT)
        raise SystemExit(1)

    requested = args.device if torch.cuda.is_available() else "cpu"
    dev = torch.device(requested if requested != "cuda" else f"cuda:{torch.cuda.current_device()}")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    device = requested
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else None

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_REPO_ROOT, "outputs", "closed_loop_benchmark", run_id)
    os.makedirs(out_dir, exist_ok=True)
    csv = clog.CSVLogger(
        os.path.join("logs", f"closed_loop_benchmark_{run_id}.csv"),
        "task", "trial", "step", "phase", "energy", "dx", "dy", "dz", "dgrip",
        "rdx", "rdy", "rdz", "ee_x", "ee_y", "ee_z", "obj_x", "obj_y", "obj_z",
        "tgt_x", "tgt_y", "tgt_z", "dist_goal_m", "obj_dz_m", "tilt_deg", "obj_speed",
        "held", "released", "cem_time_s", "success", "failure_type",
    )

    logger.info("=" * 64)
    logger.info("closed-loop smoke | tasks=%s trials=%d | samples=%d cem_steps=%d T=%d maxnorm=%.3f",
                args.tasks, args.trials, args.samples, args.cem_steps, args.rollout, args.maxnorm)
    logger.info("run_id=%s | out=%s", run_id, out_dir)
    logger.info("=" * 64)

    (encoder, predictor), load_ms = clog.gpu_timer(lambda: load_model(device))
    tokens_per_frame = (CROP // encoder.patch_size) ** 2
    logger.info("model loaded in %.1f s | tokens/frame=%d", load_ms / 1000.0, tokens_per_frame)

    _, plan = make_planner(cem, compute_new_pose, predictor, tokens_per_frame, args, dev, autocast_dtype)

    def encode_goal_factory(env, fovy):
        def _encode_goal(goal_img):
            with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                                 enabled=autocast_dtype is not None):
                return encode(encoder, goal_img, device, tokens_per_frame).detach()
        return _encode_goal

    summary = []
    for task in args.tasks:
        needs_obj = task in ("grasp_lift", "place")
        env = FrankaDroidEnv(render_width=CROP, render_height=CROP, max_translation=0.13,
                             add_object=needs_obj, add_zone=needs_obj)
        fovy = float(env.model.vis.global_.fovy)
        ctx = {
            "encoder": encoder, "plan": plan, "encode_goal": encode_goal_factory(env, fovy),
            "cem_kw": dict(encode_fn=encode, device=device, tokens_per_frame=tokens_per_frame,
                           dev=dev, autocast_dtype=autocast_dtype),
        }
        for trial in range(args.trials):
            tlog = TrialLogger(csv, task, trial, fovy, needs_obj)
            t0 = time.perf_counter()
            res = TASKS[task](env, ctx, tlog, args)
            dt = time.perf_counter() - t0
            gif = os.path.join(out_dir, f"{task}_t{trial}.gif")
            imageio.mimsave(gif, tlog.frames, fps=2, loop=0)
            save_contact_sheet(tlog.frames, os.path.join(out_dir, f"{task}_t{trial}_contact.png"))
            logger.info("[%s trial %d] success=%s failure=%s | %d frames | %.1fs -> %s",
                        task, trial, res.success, res.failure_type, len(tlog.frames), dt,
                        os.path.basename(gif))
            logger.info("    metrics: %s", {k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in res.metrics.items()})
            summary.append((task, trial, res.success, res.failure_type))
        env.close()

    logger.info("-" * 64)
    for task in args.tasks:
        rows = [s for s in summary if s[0] == task]
        ok = sum(1 for s in rows if s[2])
        logger.info("SMOKE %-11s %d/%d success | failures=%s",
                    task, ok, len(rows), [s[3] for s in rows if not s[2]])
    logger.info("artifacts: %s", out_dir)
    logger.info("done")


if __name__ == "__main__":
    main()
