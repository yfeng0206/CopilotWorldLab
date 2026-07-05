"""Closed-loop V-JEPA 2-AC benchmark runner: Reach / Grasp-Lift / Place.

The Phase-1 task-success benchmark (docs/experiments/closed_loop_benchmark.md). Runs at any scale:
1-5 trials for a smoke check, 50 trials for the full precision-curve benchmark.
For each task the model sees only an RGB image (the validated PLANNING_CAMERA), the 7-D EE state,
and a goal image; V-JEPA 2-AC plans the coarse motion with CEM MPC. Scripted primitives handle the
gripper (close/lift/open). Success is judged ONLY from hidden privileged MuJoCo truth
(src/bench/success.py) -- object pose, contacts, velocity, tilt -- never from the latent energy.

CEM config (defaults follow Meta's released wrapper; smaller population for the 3090):
    samples=200, cem_steps=10, rollout/horizon T=2, topk=10, maxnorm=0.05 m/axis, momentum 0.15.
Later ablations (not here): T=1 vs T=2, samples 200/400/800, maxnorm 0.05 vs 0.075.

Task decomposition (honest separation of V-JEPA vs scripted; success from hidden MuJoCo truth):
  reach       : pure V-JEPA closed-loop to a goal image.
  grasp_lift  : V-JEPA REACHES the grasp pose (goal image = arm at the cube); only close + lift
                are scripted -- no privileged re-centering, so the rate reflects V-JEPA's grasp.
  place       : scripted grasp to start holding (isolates placement), then V-JEPA DRIVES the held
                cube over the zone; the release lowers straight down at V-JEPA's reached xy (no
                move-to-zone-center), so the rate reflects V-JEPA's placement accuracy.

One continuous error per trial -> success@multiple precision thresholds computed from a single run.
Full run-log (config + per-step CSV + per-trial CSV + selected viz) lands under
logs/closed_loop_runs/<run_id>/; the committed report (summary.md/csv, plots, selected GIFs +
contact sheets for ~3 best/median/worst trials) lands under results/benchmarks/closed_loop_<tag>/.
Cube/target positions are randomized per trial (stably seeded; see TASK_SEED_OFFSET).

    python scripts/run_closed_loop_benchmark.py --tasks reach grasp_lift place --trials 50
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
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


def save_contact_sheet(frames, path, phases=None, cols=4):
    """Fewer, larger cells for real audit: one frame per phase transition (+ first/last), not a
    dense grid of tiny cells. Falls back to evenly-spaced frames if phases are unknown."""
    if not frames:
        return
    n = len(frames)
    if phases:
        idx = [0]
        for i in range(1, n):
            if phases[i] != phases[i - 1]:
                idx.append(i)
        idx.append(n - 1)
        idx = sorted(set(idx))
    else:
        idx = list(np.unique(np.linspace(0, n - 1, min(8, n)).astype(int)))
    rows = (len(idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.axis("off")
    for k, i in enumerate(idx):
        axes[k].imshow(frames[i])
        axes[k].set_title(f"frame {i}" + (f" ({phases[i]})" if phases else ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
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
    """Logs per-step CSV rows for one trial and stores lightweight per-step data (raw image +
    marker pixels + stats) so annotated panels can be built ON DEMAND -- only for the few trials
    selected for GIFs, keeping a 50-trial run memory-light."""

    def __init__(self, csv, task, trial, fovy, has_object):
        self.csv, self.task, self.trial, self.fovy, self.has_object = csv, task, trial, fovy, has_object
        self.steps = []  # per-step dicts: img + projected markers + stats panel (for GIF/contact)
        self.rows = []   # compact per-step dicts for the markdown frame table
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
        self.steps.append({"img": img, "obj_px": obj_px, "ee_px": ee_px,
                           "zone_px": zone_px, "zone_r_px": zone_r_px or 4.0, "stats": stats})
        self.rows.append({
            "step": self.step, "phase": phase, "energy": _fmt(energy),
            "dist_goal": _fmt(dist_goal), "obj_dz": _fmt(obj_dz), "tilt": _fmt(tilt),
            "speed": _fmt(speed), "held": held, "released": released, "cem_s": _fmt(cem_time),
        })
        self.step += 1

    def build_frames(self):
        """Build the annotated panels for this trial (only called for selected/visualized trials)."""
        return [render_panel(s["img"], s["obj_px"], s["ee_px"], s["zone_px"], s["zone_r_px"],
                             s["stats"]) for s in self.steps]

    @property
    def phases(self):
        return [r["phase"] for r in self.rows]

    def write_markdown(self, path, record, dt):
        """A readable per-step frame table (audit-friendly, unlike the compressed contact sheet)."""
        vjepa = [r for r in self.rows if str(r["phase"]).startswith("vjepa")]
        lines = [
            f"# {self.task} trial {self.trial}",
            "",
            f"- **error**: {record['error']:.4f} m  |  **failure**: {record['failure'] or '-'}",
            f"- **gates**: {record['gates']}",
            f"- **metrics**: {record['metrics']}",
            f"- V-JEPA closed-loop steps: {len(vjepa)}; total steps: {len(self.rows)}; wall {dt:.1f}s",
            "",
            "| step | phase | energy | dist_goal | obj_dz | tilt | speed | held | rel | cem_s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in self.rows:
            lines.append("| {step} | {phase} | {energy} | {dist_goal} | {obj_dz} | {tilt} | "
                         "{speed} | {held} | {released} | {cem_s} |".format(**r))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


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
    # Settle by HOLDING the current EE pose (absolute), not zero-delta: the arm position servo
    # droops under a held load, and a zero-delta command re-baselines to the drooping pose each
    # step (the arm ratchets down ~9 cm). Re-commanding the captured pose keeps it lifted.
    if settle:
        hold = env.get_ee_state()[:3].copy()
        for _ in range(settle):
            cur = env.get_ee_state()[:3]
            d = np.zeros(7)
            d[:3] = hold - cur
            env.apply_action(d)
    tlog.record(env, phase, None, None, float("nan"),
                target if target is not None else None, float("nan"), float("nan"))


# ----------------------------------------------------------------------------- tasks
# Precision-curve thresholds (metres): success is evaluated at MANY thresholds from one rollout
# (the recorded continuous error), so we get a precision curve instead of one arbitrary cutoff.
THRESHOLDS = {
    "reach": [0.05, 0.03, 0.015],
    "grasp_lift": [0.06, 0.05, 0.03, 0.02],
    "place": [0.10, 0.06, 0.03, 0.015],
}
# The physical gates that must ALSO hold (beyond the precision threshold) for a real success.
GATE_SPEC = {
    "reach": [],
    "grasp_lift": ["lifted", "held", "upright", "stable"],
    "place": ["upright", "stable", "released"],
}
# Stable per-task seed offsets. Do NOT use hash(task): Python randomizes string hashing per
# process (PYTHONHASHSEED), so it would break cross-run reproducibility of the seeded init.
TASK_SEED_OFFSET = {"reach": 0, "grasp_lift": 1, "place": 2}


def _rand_cube_xy(rng):
    """A randomized, reachable cube start on the table (around CUBE_START)."""
    return (float(rng.uniform(0.45, 0.55)), float(rng.uniform(-0.15, -0.05)))


def task_reach(env, ctx, tlog, args):
    rng = ctx["rng"]
    env.reset()
    home = env.get_ee_state()[:3].copy()
    target = home + np.array([float(rng.uniform(0.06, 0.12)), float(rng.uniform(-0.10, 0.02)),
                              float(rng.uniform(-0.08, -0.02))])
    goal_img = env.capture_goal_image(pos=target, euler=EE_DOWN, camera="planning")
    z_goal = ctx["encode_goal"](goal_img)
    env.reset()
    steps, dist = cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, target, tlog,
                              "vjepa_reach", args.reach_steps, args.pos_tol, **ctx["cem_kw"])
    error = float(np.linalg.norm(env.get_ee_state()[:3] - target))
    res = reach_success(env.get_ee_state()[:3], target, tau_reach=max(THRESHOLDS["reach"]))
    tlog.record(env, "final", None, None, float("nan"), target, float("nan"), error,
                success=int(res.success), failure=res.failure_type or "")
    return {"error": error, "gates": {}, "failure": res.failure_type or "", "metrics": res.metrics}


def _scripted_grasp(env, tlog):
    """Reliable scripted grasp primitive (used to START the PLACE task holding the cube): align
    above -> descend -> close (re-seating the fingers) -> lift -> settle. Returns object z before
    the lift. Uses privileged cube xy -- appropriate only for isolating the downstream skill
    (placement), NOT for the grasp_lift task, which tests V-JEPA's own grasp reach."""
    c = env.object_position()
    scripted(env, tlog, "align", [c[0], c[1], c[2] + 0.12])
    c = env.object_position()
    grasp = [c[0], c[1], c[2] + 0.005]
    scripted(env, tlog, "descend", grasp)
    scripted(env, tlog, "close", grasp, gripper=1.0)
    obj_z0 = float(env.object_position()[2])
    scripted(env, tlog, "lift", None, dz=0.12)
    scripted(env, tlog, "settle", None, settle=20)
    return obj_z0


def task_grasp_lift(env, ctx, tlog, args):
    """V-JEPA must REACH the grasp pose (goal image = the arm at the cube, grasp-ready). Only the
    close + lift are scripted -- no privileged re-centering -- so the metrics reflect V-JEPA's own
    grasp-positioning ability. Precision error = ||object_xy - EE_xy|| BEFORE the close."""
    cube_xy = _rand_cube_xy(ctx["rng"])
    env.reset(cube_xy=cube_xy)
    c = env.object_position()
    grasp_pos = np.array([c[0], c[1], c[2] + 0.005])  # AT the cube (fingers around it), open
    goal_img = env.capture_goal_image(pos=grasp_pos, euler=EE_DOWN, gripper=0.0, camera="planning")
    z_goal = ctx["encode_goal"](goal_img)
    env.reset(cube_xy=cube_xy)
    # 1) V-JEPA closed-loop reach to the grasp pose (gripper frozen open the whole way)
    cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, grasp_pos, tlog,
                "vjepa_reach", args.grasp_steps, args.pos_tol, **ctx["cem_kw"])
    # precision error: how well V-JEPA positioned the gripper over the object BEFORE the close
    grasp_xy_error = float(np.linalg.norm(env.object_position()[:2] - env.get_ee_state()[:2]))
    # 2) ONLY scripted: close (holding the pose V-JEPA reached) then lift -- no re-centering.
    obj_z0 = float(env.object_position()[2])
    scripted(env, tlog, "close", None, gripper=1.0)
    scripted(env, tlog, "lift", None, dz=0.12)
    scripted(env, tlog, "settle", None, settle=20)
    obj = env.object_position()
    spec = SUCCESS_DEFAULTS["grasp_lift"]
    res = grasp_lift_success(obj_z0, float(obj[2]), env.get_ee_state()[:2], obj[:2],
                             env.object_tilt(), env.object_speed(), env.gripper_holds_object(), spec)
    gates = {
        "lifted": float(obj[2] - obj_z0) > spec["lift_dz"],
        "held": bool(env.gripper_holds_object()),
        "upright": float(np.degrees(env.object_tilt())) < spec["tilt_max_deg"],
        "stable": float(env.object_speed()) < spec["v_settle"],
    }
    tlog.record(env, "final", None, None, float("nan"), grasp_pos, float("nan"), grasp_xy_error,
                success=int(res.success), failure=res.failure_type or "")
    return {"error": grasp_xy_error, "gates": gates, "failure": res.failure_type or "",
            "metrics": res.metrics}


def task_place(env, ctx, tlog, args):
    """Start holding the cube via the reliable scripted grasp, then V-JEPA must DRIVE the held cube
    over the zone. The final descent is straight DOWN at whatever xy V-JEPA reached (no scripted
    move-to-zone-center), so placement error = ||object_xy_final - zone_xy|| reflects V-JEPA's own
    horizontal accuracy."""
    cube_xy = _rand_cube_xy(ctx["rng"])
    env.reset(cube_xy=cube_xy)
    _scripted_grasp(env, tlog)  # reliable grasp to isolate the PLACEMENT skill
    if not env.gripper_holds_object():
        err = float(np.linalg.norm(env.object_position()[:2] - env.zone_center()))
        tlog.record(env, "final", None, None, float("nan"), None, float("nan"), err,
                    success=0, failure="grasp_failed_pre_place")
        return {"error": err, "gates": {g: False for g in GATE_SPEC["place"]},
                "failure": "grasp_failed_pre_place", "metrics": {}}
    # V-JEPA reach: drive the held cube to a hover over the zone
    zone = env.zone_center()
    place_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.12])
    goal_img = env.capture_goal_image(pos=place_pos, euler=EE_DOWN, gripper=1.0, camera="planning")
    z_goal = ctx["encode_goal"](goal_img)
    cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, place_pos, tlog,
                "vjepa_place", args.place_steps, args.pos_tol, **ctx["cem_kw"])
    # scripted release: lower STRAIGHT DOWN at V-JEPA's reached xy (no move-to-zone-center), open
    cur = env.get_ee_state()[:3]
    scripted(env, tlog, "lower", [cur[0], cur[1], TABLE_TOP_Z + CUBE_HALF + 0.02])
    scripted(env, tlog, "open", None, gripper=-1.0)
    scripted(env, tlog, "settle", None, settle=30)
    obj = env.object_position()
    place_xy_error = float(np.linalg.norm(obj[:2] - env.zone_center()))
    spec = SUCCESS_DEFAULTS["place"]
    res = place_success(obj[:2], env.zone_center(), env.object_tilt(), env.object_speed(),
                        env.object_released(), spec)
    gates = {
        "upright": float(np.degrees(env.object_tilt())) < spec["tilt_max_deg"],
        "stable": float(env.object_speed()) < spec["v_settle"],
        "released": bool(env.object_released()),
    }
    tlog.record(env, "final", None, None, float("nan"), place_pos, float("nan"), place_xy_error,
                success=int(res.success), failure=res.failure_type or "")
    return {"error": place_xy_error, "gates": gates, "failure": res.failure_type or "",
            "metrics": res.metrics}


TASKS = {"reach": task_reach, "grasp_lift": task_grasp_lift, "place": task_place}


# ----------------------------------------------------------------------------- summary / provenance
def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return ""


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _pf(s):
    """Parse a formatted stat string (e.g. '+0.1234' or '-') to float or None."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _cm(t):
    """Format a metre threshold as a cm label (e.g. 0.015 -> '1.5', 0.05 -> '5')."""
    return f"{t * 100:g}"


def success_at(error, gates, thr):
    """A trial is a success at precision `thr` iff error < thr AND all physical gates hold."""
    return bool(error < thr and all(gates.values()))


def select_for_viz(records):
    """Indices of ~3 best, ~3 median, ~3 worst trials by error (deduplicated), for GIFs."""
    order = sorted(range(len(records)), key=lambda i: records[i]["error"])
    n = len(order)
    if n <= 9:
        return order
    mid = n // 2
    picks = order[:3] + order[mid - 1:mid + 2] + order[-3:]
    return sorted(set(picks))


def summarize_task(records, task):
    errs = np.array([r["error"] for r in records], dtype=float)
    fails = {}
    for r in records:
        if not r["all_success"]:
            fails[r["failure"] or "unknown"] = fails.get(r["failure"] or "unknown", 0) + 1
    row = {
        "task": task, "n": len(records),
        "mean_err": float(errs.mean()), "median_err": float(np.median(errs)),
        "p75_err": float(np.percentile(errs, 75)), "p90_err": float(np.percentile(errs, 90)),
        "mean_steps": float(np.mean([r["total_steps"] for r in records])),
        "mean_vjepa_steps": float(np.mean([r["vjepa_steps"] for r in records])),
        "mean_cem_s": float(np.nanmean([r["mean_cem_s"] for r in records])),
        "failures": fails,
    }
    for thr in THRESHOLDS[task]:
        row[f"succ@{thr}"] = float(np.mean([success_at(r["error"], r["gates"], thr)
                                            for r in records]))
    return row


def write_task_plots(records, task, path):
    errs = np.array([r["error"] for r in records], dtype=float)
    thrs = THRESHOLDS[task]
    rates = [np.mean([success_at(r["error"], r["gates"], t) for r in records]) for t in thrs]
    fails = {}
    for r in records:
        if not r["all_success"]:
            fails[r["failure"] or "unknown"] = fails.get(r["failure"] or "unknown", 0) + 1
    energies = [r["final_energy"] for r in records if r["final_energy"] is not None]
    energy_err = [(r["final_energy"], r["error"]) for r in records if r["final_energy"] is not None]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(errs * 100, bins=15, color="#3b7dd8", edgecolor="white")
    ax[0, 0].set_xlabel("final error (cm)")
    ax[0, 0].set_ylabel("trials")
    ax[0, 0].set_title(f"{task}: error distribution (n={len(records)})")

    ax[0, 1].plot([t * 100 for t in thrs], [r * 100 for r in rates], "o-", color="#d1495b")
    for t, r in zip(thrs, rates):
        ax[0, 1].annotate(f"{r*100:.0f}%", (t * 100, r * 100), fontsize=8,
                          textcoords="offset points", xytext=(0, 6))
    ax[0, 1].set_xlabel("precision threshold (cm)")
    ax[0, 1].set_ylabel("success rate (%)")
    ax[0, 1].set_ylim(-5, 105)
    ax[0, 1].set_title("success vs threshold (precision curve)")

    if fails:
        ax[1, 0].bar(list(fails.keys()), list(fails.values()), color="#e08a3c")
        ax[1, 0].set_ylabel("count")
        ax[1, 0].tick_params(axis="x", rotation=30, labelsize=8)
    ax[1, 0].set_title("failure types")

    if energy_err:
        e, er = zip(*energy_err)
        ax[1, 1].scatter(e, [x * 100 for x in er], color="#5a5a5a", alpha=0.7)
        ax[1, 1].set_xlabel("final latent energy")
        ax[1, 1].set_ylabel("final error (cm)")
    ax[1, 1].set_title("error vs final latent energy")

    fig.suptitle(f"Closed-loop benchmark -- {task}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    import torch

    from utils.mpc_utils import cem, compute_new_pose

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["reach", "grasp_lift", "place"], choices=list(TASKS))
    p.add_argument("--trials", type=int, default=1, help="trials per task (smoke: 1-5; full: 50)")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--cem-steps", type=int, default=10)
    p.add_argument("--rollout", type=int, default=2, help="CEM planning horizon T")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--maxnorm", type=float, default=0.05, help="CEM per-axis action clip (m)")
    p.add_argument("--pos-tol", type=float, default=0.03, help="reached if EE within this of goal (m)")
    p.add_argument("--reach-steps", type=int, default=5)
    p.add_argument("--grasp-steps", type=int, default=6)
    p.add_argument("--place-steps", type=int, default=8)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help="seed for per-trial randomized init")
    p.add_argument("--tag", default="full", help="report subdir: results/benchmarks/closed_loop_<tag>")
    p.add_argument("--viz-only-selected", action="store_true", default=True,
                   help="save GIFs only for ~3 best/median/worst trials (default on)")
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
    # Comprehensive, self-contained RUN LOG dir (gitignored under logs/) for later diagnosis:
    # full config + every per-step row. The committed report (summary/plots/selected GIFs) is
    # written to results/benchmarks/closed_loop_<tag>/<run_id>/.
    run_dir = os.path.join(_REPO_ROOT, "logs", "closed_loop_runs", run_id)
    viz_dir = os.path.join(run_dir, "viz")
    report_dir = os.path.join(_REPO_ROOT, "results", "benchmarks", f"closed_loop_{args.tag}", run_id)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    csv = clog.CSVLogger(
        os.path.join(run_dir, "steps.csv"),
        "task", "trial", "step", "phase", "energy", "dx", "dy", "dz", "dgrip",
        "rdx", "rdy", "rdz", "ee_x", "ee_y", "ee_z", "obj_x", "obj_y", "obj_z",
        "tgt_x", "tgt_y", "tgt_z", "err_m", "obj_dz_m", "tilt_deg", "obj_speed",
        "held", "released", "cem_time_s", "success", "failure_type",
    )

    config = {
        "run_id": run_id, "git_commit": _git_commit(), "timestamp": datetime.now().isoformat(),
        "model": "vjepa2-ac-vitg (ViT-g encoder 1.01B + AC predictor 305M)",
        "checkpoint": os.path.relpath(CHECKPOINT, _REPO_ROOT), "checkpoint_sha256": _sha256(CHECKPOINT),
        "device": str(dev), "dtype": args.dtype,
        "cem": {"samples": args.samples, "cem_steps": args.cem_steps, "rollout_T": args.rollout,
                "topk": args.topk, "maxnorm": args.maxnorm, "momentum_mean": 0.15,
                "objective": "mean-L1 in layer-norm'd latent, gripper axis frozen"},
        "tasks": args.tasks, "trials_per_task": args.trials, "seed": args.seed,
        "pos_tol": args.pos_tol,
        "max_vjepa_steps": {"reach": args.reach_steps, "grasp_lift": args.grasp_steps,
                            "place": args.place_steps},
        "thresholds_m": THRESHOLDS, "gate_spec": GATE_SPEC,
        "env": {"embodiment": "Franka Panda + Robotiq 2F-85 (FrankaDroidEnv)",
                "camera": "PLANNING_CAMERA (az45_el45 free cam)", "crop": CROP,
                "cube_half_m": CUBE_HALF, "cube_start_randomized": True,
                "zone_radius_m": SUCCESS_DEFAULTS["place"]["zone_radius"]},
        "normalization": "ImageNet mean/std x255 on 0-255 input (matches vendored make_transforms)",
        "logs": {"run_dir": os.path.relpath(run_dir, _REPO_ROOT),
                 "steps_csv": "steps.csv", "trials_csv": "trials.csv",
                 "report_dir": os.path.relpath(report_dir, _REPO_ROOT)},
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 64)
    logger.info("closed-loop benchmark | tasks=%s trials=%d | samples=%d cem_steps=%d T=%d maxnorm=%.3f",
                args.tasks, args.trials, args.samples, args.cem_steps, args.rollout, args.maxnorm)
    logger.info("run_id=%s | run_log=%s | report=%s", run_id,
                os.path.relpath(run_dir, _REPO_ROOT), os.path.relpath(report_dir, _REPO_ROOT))
    logger.info("=" * 64)

    (encoder, predictor), load_ms = clog.gpu_timer(lambda: load_model(device))
    tokens_per_frame = (CROP // encoder.patch_size) ** 2
    logger.info("model loaded in %.1f s | tokens/frame=%d", load_ms / 1000.0, tokens_per_frame)

    _, plan = make_planner(cem, compute_new_pose, predictor, tokens_per_frame, args, dev, autocast_dtype)

    def encode_goal_factory():
        def _encode_goal(goal_img):
            with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                                 enabled=autocast_dtype is not None):
                return encode(encoder, goal_img, device, tokens_per_frame).detach()
        return _encode_goal

    trial_csv = clog.CSVLogger(
        os.path.join(run_dir, "trials.csv"),
        "task", "trial", "error_m", "success_loose", "failure", "final_energy",
        "vjepa_steps", "total_steps", "mean_cem_s", "wall_s", "success_at_thresholds",
    )

    all_records = {}
    for task in args.tasks:
        needs_obj = task in ("grasp_lift", "place")
        env = FrankaDroidEnv(render_width=CROP, render_height=CROP, max_translation=0.13,
                             add_object=needs_obj, add_zone=needs_obj)
        fovy = float(env.model.vis.global_.fovy)
        task_off = TASK_SEED_OFFSET[task] * 100003
        ctx = {"encoder": encoder, "plan": plan, "encode_goal": encode_goal_factory(),
               "cem_kw": dict(encode_fn=encode, device=device, tokens_per_frame=tokens_per_frame,
                              dev=dev, autocast_dtype=autocast_dtype)}
        records, tlogs = [], []
        for trial in range(args.trials):
            # deterministic per-trial seeding (numpy init + torch CEM sampling) for reproducibility
            ctx["rng"] = np.random.default_rng(args.seed * 100003 + task_off + trial)
            torch.manual_seed(args.seed * 100003 + task_off + trial)
            tlog = TrialLogger(csv, task, trial, fovy, needs_obj)
            t0 = time.perf_counter()
            rec = TASKS[task](env, ctx, tlog, args)
            dt = time.perf_counter() - t0
            vjepa_rows = [r for r in tlog.rows if str(r["phase"]).startswith("vjepa")]
            energies = [_pf(r["energy"]) for r in vjepa_rows if _pf(r["energy"]) is not None]
            cems = [_pf(r["cem_s"]) for r in tlog.rows if _pf(r["cem_s"]) is not None]
            thr_flags = {t: success_at(rec["error"], rec["gates"], t) for t in THRESHOLDS[task]}
            rec.update(task=task, trial=trial, wall_s=round(dt, 1),
                       final_energy=(energies[-1] if energies else None),
                       vjepa_steps=len(vjepa_rows), total_steps=len(tlog.rows),
                       mean_cem_s=(float(np.mean(cems)) if cems else float("nan")),
                       thr_flags=thr_flags,
                       all_success=any(thr_flags.values()))  # success at the LOOSEST threshold
            records.append(rec)
            tlogs.append(tlog)
            trial_csv.log(task, trial, round(rec["error"], 4), int(rec["all_success"]),
                          rec["failure"], _r(rec["final_energy"]), rec["vjepa_steps"],
                          rec["total_steps"], _r(rec["mean_cem_s"]), rec["wall_s"],
                          json.dumps({str(k): int(v) for k, v in thr_flags.items()}))
            logger.info("[%s t%02d] error=%.4f m gates=%s failure=%s | vjepa=%d steps | %.0fs",
                        task, trial, rec["error"], rec["gates"], rec["failure"] or "-",
                        rec["vjepa_steps"], dt)
        env.close()

        # visualize only the selected (best/median/worst) trials -> GIF + contact sheet + md
        sel = select_for_viz(records)
        for i in sel:
            frames = tlogs[i].build_frames()
            imageio.mimsave(os.path.join(viz_dir, f"{task}_t{i}.gif"), frames, fps=2, loop=0)
            save_contact_sheet(frames, os.path.join(viz_dir, f"{task}_t{i}_contact.png"),
                               phases=tlogs[i].phases)
            tlogs[i].write_markdown(os.path.join(viz_dir, f"{task}_t{i}_frames.md"), records[i],
                                    records[i]["wall_s"])
        write_task_plots(records, task, os.path.join(report_dir, f"{task}_summary.png"))
        all_records[task] = records

    # ---- summary tables (committed) + copy selected GIFs/contact sheets into the report ----
    summaries = [summarize_task(all_records[t], t) for t in args.tasks]
    _write_summary(report_dir, run_id, config, summaries, args)
    for task in args.tasks:
        for i in select_for_viz(all_records[task]):
            for suf in (f"{task}_t{i}.gif", f"{task}_t{i}_contact.png"):
                src = os.path.join(viz_dir, suf)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(report_dir, suf))

    logger.info("-" * 64)
    logger.info("PRECISION-CURVE SUMMARY (samples=%d T=%d cem_steps=%d maxnorm=%.3f seed=%d)",
                args.samples, args.rollout, args.cem_steps, args.maxnorm, args.seed)
    for s in summaries:
        thr_str = " ".join(f"@{_cm(t)}cm={s[f'succ@{t}']*100:.0f}%" for t in THRESHOLDS[s["task"]])
        logger.info("  %-11s n=%d mean_err=%.3f median=%.3f p90=%.3f | %s",
                    s["task"], s["n"], s["mean_err"], s["median_err"], s["p90_err"], thr_str)
    logger.info("run log (gitignored): %s", os.path.relpath(run_dir, _REPO_ROOT))
    logger.info("committed report: %s", os.path.relpath(report_dir, _REPO_ROOT))
    logger.info("done")


def _write_summary(report_dir, run_id, config, summaries, args):
    """Write summary.csv + summary.md into the committed report dir."""
    scsv = clog.CSVLogger(
        os.path.join(report_dir, "summary.csv"),
        "task", "n", "mean_err_m", "median_err_m", "p75_err_m", "p90_err_m",
        "mean_steps", "mean_vjepa_steps", "mean_cem_s", "thresholds_m", "success_rates",
    )
    for s in summaries:
        rates = {f"{t}": round(s[f"succ@{t}"], 3) for t in THRESHOLDS[s["task"]]}
        scsv.log(s["task"], s["n"], round(s["mean_err"], 4), round(s["median_err"], 4),
                 round(s["p75_err"], 4), round(s["p90_err"], 4), round(s["mean_steps"], 1),
                 round(s["mean_vjepa_steps"], 1), round(s["mean_cem_s"], 2),
                 str(THRESHOLDS[s["task"]]), json.dumps(rates))

    lines = [
        f"# Closed-loop benchmark -- run {run_id}",
        "",
        f"Config: model **{config['model']}**, samples **{args.samples}**, cem_steps "
        f"**{args.cem_steps}**, rollout **T={args.rollout}**, topk **{args.topk}**, maxnorm "
        f"**{args.maxnorm} m**, dtype **{args.dtype}**, **{args.trials} trials/task**, seed {args.seed}.",
        f"Commit `{config['git_commit'][:10]}`. Success = error < threshold AND physical gates "
        "(lifted/held/upright/stable/released), judged from hidden MuJoCo truth. Cube/target "
        "positions randomized per trial.",
        "",
        "## Precision curve (success rate at multiple thresholds, one run)",
        "",
    ]
    for s in summaries:
        thrs = THRESHOLDS[s["task"]]
        header = "| task | n | mean err (cm) | median | p90 | " + \
                 " | ".join(f"@{_cm(t)}cm" for t in thrs) + " |"
        sep = "|" + "---|" * (5 + len(thrs))
        row = (f"| **{s['task']}** | {s['n']} | {s['mean_err']*100:.1f} | "
               f"{s['median_err']*100:.1f} | {s['p90_err']*100:.1f} | "
               + " | ".join(f"{s[f'succ@{t}']*100:.0f}%" for t in thrs) + " |")
        lines += [header, sep, row, ""]
        if s["failures"]:
            lines.append(f"- {s['task']} failures: {s['failures']}")
        lines.append(f"- {s['task']} mean steps {s['mean_steps']:.1f} "
                     f"(V-JEPA {s['mean_vjepa_steps']:.1f}), mean CEM {s['mean_cem_s']:.1f} s/step")
        lines.append("")
    lines += [
        "## Task decomposition (what V-JEPA does vs scripted)",
        "- **reach**: pure V-JEPA closed-loop to a goal image.",
        "- **grasp_lift**: V-JEPA reaches the grasp pose; only close+lift scripted "
        "(error = object-EE xy before close).",
        "- **place**: scripted grasp, then V-JEPA drives the held cube over the zone; release "
        "lowers straight down (error = object-zone xy).",
        "",
        "Plots: `<task>_summary.png` (error histogram, precision curve, failure types, "
        "error-vs-energy). Selected GIFs/contact sheets: 3 best/median/worst per task. Full "
        "per-step logs + config: gitignored `logs/closed_loop_runs/<run_id>/`.",
    ]
    with open(os.path.join(report_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
