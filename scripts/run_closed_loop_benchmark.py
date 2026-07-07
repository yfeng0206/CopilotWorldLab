"""Closed-loop V-JEPA 2-AC benchmark runner (fixed-bundle + legacy modes).

The Phase-1 task-success benchmark (docs/experiments/closed_loop_benchmark.md). Runs at any scale:
1-5 trials for a smoke check, 50 trials for the full precision-curve benchmark.
For each task the model sees only an RGB image (the validated PLANNING_CAMERA), the 7-D EE state,
and a goal image; V-JEPA 2-AC plans the coarse motion with CEM MPC. Scripted primitives handle the
gripper (close/lift/open). Success is judged ONLY from hidden privileged MuJoCo truth
(src/bench/success.py) -- object pose, contacts, velocity, tilt -- never from the latent energy.

Two modes:
  --bundles <dir>  FIXED-BUNDLE mode (primary): load the saved, inspectable task bundles from
                   <dir>/<task>/<object>/ (see scripts/generate_task_bundles.py) and score every
                   config on identical scenarios. Tasks: grasp / reach_with_object / grasp_and_reach
                   / pick_place, each per object (cup / box). Each trial restores the exact recorded
                   start (env.set_state(qpos0) + mocap zone, honoring start_grasped), plans to the
                   SAVED goal/sub-goal images (pick_place on the paper 4/10/4 schedule), scripts only
                   the gripper, and scores from hidden state with the swept sphere x.
  (legacy)         random-per-trial reach / grasp_lift / place / pick_place (stably seeded; see
                   TASK_SEED_OFFSET). Used before the fixed bundles existed.

CEM config (paper p.37: 800 samples, 10 refinement steps, top-10, planning horizon 1; maxnorm ~0.075
per section 4.1). Defaults here use the paper's horizon T=1; samples/maxnorm are swept:
    samples=200 (paper 800), cem_steps=10, horizon T=1, topk=10, maxnorm=0.05 m/axis (paper ~0.075),
    momentum 0.15.

One continuous error per trial -> success@multiple precision thresholds computed from a single run.
Full run-log (config + per-step CSV + per-trial CSV + selected viz) lands under
logs/closed_loop_runs/<run_id>/; the committed report (summary.md/csv, plots, selected GIFs +
contact sheets for ~3 best/median/worst trials) lands under results/benchmarks/closed_loop_<tag>/.

    python scripts/run_closed_loop_benchmark.py --bundles tasks \\
        --tasks grasp reach_with_object grasp_and_reach pick_place --objects cup box --trials 50
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
import zlib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

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
    OBJECT_SPECS,
    PLANNING_CAMERA,
    TABLE_TOP_Z,
)
from src.envs.franka_droid_env import FrankaDroidEnv  # noqa: E402
from src.bench.schema import SUCCESS_DEFAULTS, TaskBundle  # noqa: E402
from src.bench.success import (  # noqa: E402
    bundle_classify,
    grasp_lift_success,
    place_success,
    reach_success,
)
from src.bench.thresholds import (  # noqa: E402
    BUNDLE_TASKS,
    GATE_SPEC,
    LEGACY_TASKS,
    THRESHOLDS,
    success_at,
    validate_task_mode,
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


def _demo_panel(step, label, sub_i, sub_n, dist):
    """One demo frame for a single rollout: the camera image with red/blue/green markers, a title
    band (label), and a small caption (step index + distance-to-goal). Returns an RGB uint8 array."""
    img = step["img"]
    fig = plt.figure(figsize=(3.2, 3.5), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.90])
    ax.imshow(img)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    if step.get("zone_px") is not None and np.isfinite(step["zone_px"]).all():
        ax.add_patch(plt.Circle((step["zone_px"][0], step["zone_px"][1]),
                                 max(step.get("zone_r_px", 4.0), 3.0), fill=False, color="lime", lw=2.0))
    if step.get("ee_px") is not None and np.isfinite(step["ee_px"]).all():
        ax.plot(step["ee_px"][0], step["ee_px"][1], "o", color="blue", ms=9, mec="white", mew=1.2)
    if step.get("obj_px") is not None and np.isfinite(step["obj_px"]).all():
        ax.plot(step["obj_px"][0], step["obj_px"][1], "o", color="red", ms=9, mec="white", mew=1.2)
    cap = f"step {sub_i}/{sub_n}" + (f"   dist {dist:.3f} m" if dist is not None and np.isfinite(dist) else "")
    fig.text(0.5, 0.955, label, ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.045, cap, ha="center", va="center", fontsize=9, family="monospace")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return arr


def side_by_side_gif(left_steps, right_steps, path, label_left, label_right, fps=4):
    """Stitch two rollouts into one side-by-side comparison GIF (left | right), padding the shorter
    rollout by holding its final frame so both play in sync. Each side shows the camera view with
    markers, a title band, and a step/distance caption."""
    n = max(len(left_steps), len(right_steps))
    frames = []
    for i in range(n):
        li = min(i, len(left_steps) - 1)
        ri = min(i, len(right_steps) - 1)
        ls, rs = left_steps[li], right_steps[ri]
        left = _demo_panel(ls, label_left, li, len(left_steps) - 1, ls["stats_dist"])
        right = _demo_panel(rs, label_right, ri, len(right_steps) - 1, rs["stats_dist"])
        h = max(left.shape[0], right.shape[0])
        div = np.full((h, 4, 3), 40, dtype=np.uint8)
        frames.append(np.hstack([left, div, right]))
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


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
        self.record_qpos = False   # set True (via --replay-record) to capture qpos for 3D replay
        self.qpos_frames = []      # per-step full qpos (for the interactive MuJoCo rollout viewer)
        self.qpos_stats = []       # per-step raw scalars aligned with qpos_frames

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
            rest_half = float(getattr(env, "object_rest_half_z", CUBE_HALF))
            obj_dz = float(obj[2] - (TABLE_TOP_Z + rest_half))
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
                           "zone_px": zone_px, "zone_r_px": zone_r_px or 4.0, "stats": stats,
                           "stats_dist": float(dist_goal) if dist_goal is not None else float("nan")})
        self.rows.append({
            "step": self.step, "phase": phase, "energy": _fmt(energy),
            "dist_goal": _fmt(dist_goal), "obj_dz": _fmt(obj_dz), "tilt": _fmt(tilt),
            "speed": _fmt(speed), "held": held, "released": released, "cem_s": _fmt(cem_time),
        })
        if self.record_qpos:
            def _f(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return float("nan")
            self.qpos_frames.append(env.data.qpos.copy())
            self.qpos_stats.append({"phase": phase, "step": self.step, "dist": _f(dist_goal),
                                    "energy": _f(energy), "tilt": _f(tilt),
                                    "held": (int(held) if held != "" else -1), "grip": _f(a[6])})
        self.step += 1

    def save_rollout(self, path, object_type):
        """Save the recorded per-step qpos + scalars for 3D replay (scripts/replay_rollout_viewer.py)."""
        st = self.qpos_stats
        np.savez(path,
                 qpos=np.asarray(self.qpos_frames, dtype=float),
                 phase=np.asarray([s["phase"] for s in st]),
                 dist=np.asarray([s["dist"] for s in st], dtype=float),
                 energy=np.asarray([s["energy"] for s in st], dtype=float),
                 tilt=np.asarray([s["tilt"] for s in st], dtype=float),
                 held=np.asarray([s["held"] for s in st], dtype=float),
                 grip=np.asarray([s["grip"] for s in st], dtype=float),
                 object_type=object_type, task=self.task, trial=int(self.trial))

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
        reps_flat = reps.flatten(1, 2)
        # Chunk the predictor forward over the CEM sample batch (samples are independent, so this is
        # mathematically identical) to cap peak VRAM. args.chunk<=0 or >=b runs the whole batch.
        chunk = args.chunk if (0 < args.chunk < b) else b
        if chunk >= b:
            nxt = predictor(reps_flat, actions, poses)[:, -tokens_per_frame:]
        else:
            outs = []
            for i in range(0, b, chunk):
                sl = slice(i, i + chunk)
                outs.append(predictor(reps_flat[sl], actions[sl], poses[sl])[:, -tokens_per_frame:])
            nxt = torch.cat(outs, dim=0)
        nxt = F.layer_norm(nxt, (nxt.size(-1),)).view(b, 1, n_t, d)
        return nxt, compute_new_pose(poses[:, -1:], actions[:, -1:])

    def plan(z_ctx, s_ctx, z_goal):
        with torch.no_grad(), torch.autocast(dev.type, dtype=autocast_dtype,
                                             enabled=autocast_dtype is not None):
            return cem(context_frame=z_ctx, context_pose=s_ctx, goal_frame=z_goal,
                       world_model=step_predictor, rollout=args.rollout, samples=args.samples,
                       cem_steps=args.cem_steps, topk=args.topk, maxnorm=args.maxnorm,
                       axis=({} if getattr(args, "plan_gripper", False) else {3: 0.0}),
                       momentum_mean=0.15, momentum_std=0.15,
                       momentum_mean_gripper=0.15, momentum_std_gripper=0.15)

    return step_predictor, plan


def cem_to_goal(env, encoder, predictor_plan, z_goal, target, tlog, phase, max_steps,
                pos_tol, encode_fn, device, tokens_per_frame, dev, autocast_dtype,
                freeze_gripper=True):
    """Closed-loop CEM to a goal image. Returns steps taken and final distance.

    ``freeze_gripper`` (default True) zeros the executed gripper action so V-JEPA plans arm-only and
    the gripper is scripted afterward (our honest-separation baseline). Set False (via
    ``--plan-gripper``) to execute the gripper action the CEM chose -- the paper-faithful mode, where
    the goal images (which show a closed/open gripper) become reachable by the planner itself."""
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
        if freeze_gripper:
            action[6] = 0.0  # gripper frozen during the V-JEPA reach (scripted close/open afterward)
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


# ----------------------------------------------------------------------------- stage abstraction
@dataclass
class Stage:
    """One V-JEPA closed-loop sub-goal. ``goal_fn(env, ctx) -> (goal_img, target_xyz)`` lazily
    builds the goal image (previewed just before the stage runs, so sequential sub-goals see the
    current state); ``max_steps`` is this stage's CEM budget. Multiple stages express the paper's
    long-horizon sub-goal schedule (e.g. pregrasp -> grasp, or grasp -> vicinity -> place);
    a single stage is the single-goal protocol. ``name`` must start with ``vjepa`` so step/energy
    accounting picks it up. ``fixed_steps`` runs exactly ``max_steps`` with no distance early-stop
    (matches the paper's fixed 4/10/4 pick-and-place schedule, which switches sub-goals on a step
    budget, not on reaching)."""
    name: str
    goal_fn: Callable
    max_steps: int
    fixed_steps: bool = False


def run_vjepa_stages(env, ctx, tlog, stages, args) -> Optional[np.ndarray]:
    """Run a sequence of V-JEPA sub-goals with single-goal CEM per stage, swapping the goal image
    between stages (this is how the released single-``goal_frame`` CEM realizes multi-sub-goal
    planning). Returns the final stage's target xyz (used by reach for its error)."""
    target = None
    for st in stages:
        goal_img, target = st.goal_fn(env, ctx)
        z_goal = ctx["encode_goal"](goal_img)
        # fixed_steps: pass a pos_tol that never triggers, so the stage runs its full step budget
        pos_tol = -1.0 if st.fixed_steps else args.pos_tol
        cem_to_goal(env, ctx["encoder"], ctx["plan"], z_goal, target, tlog,
                    st.name, st.max_steps, pos_tol, **ctx["cem_kw"])
    return target


# ----------------------------------------------------------------------------- tasks
# THRESHOLDS + GATE_SPEC + success_at live in src/bench/thresholds.py (importable + unit-tested).
# Stable per-task seed offsets. Do NOT use hash(task): Python randomizes string hashing per
# process (PYTHONHASHSEED), so it would break cross-run reproducibility of the seeded init.
TASK_SEED_OFFSET = {"reach": 0, "grasp_lift": 1, "place": 2, "pick_place": 3}


def _rand_cube_xy(rng):
    """A randomized, reachable cube start on the table (around CUBE_START)."""
    return (float(rng.uniform(0.45, 0.55)), float(rng.uniform(-0.15, -0.05)))


def task_reach(env, ctx, tlog, args):
    """Pure V-JEPA closed-loop to a goal image (identical under both protocols: a single stage)."""
    target = _reach_target(env, ctx["rng"])
    env.reset()

    def reach_goal(e, ctx):
        return e.capture_goal_image(pos=target, euler=EE_DOWN, camera="planning"), target

    run_vjepa_stages(env, ctx, tlog, [Stage("vjepa_reach", reach_goal, args.reach_steps)], args)
    error = float(np.linalg.norm(env.get_ee_state()[:3] - target))
    res = reach_success(env.get_ee_state()[:3], target, tau_reach=max(THRESHOLDS["reach"]))
    tlog.record(env, "final", None, None, float("nan"), target, float("nan"), error,
                success=int(res.success), failure=res.failure_type or "")
    return {"error": error, "gates": {}, "failure": res.failure_type or "", "metrics": res.metrics}


def _reach_target(env, rng):
    """The seeded reach target used by both task_reach and the demo (home + a small offset)."""
    env.reset()
    home = env.get_ee_state()[:3].copy()
    return home + np.array([float(rng.uniform(0.06, 0.12)), float(rng.uniform(-0.10, 0.02)),
                            float(rng.uniform(-0.08, -0.02))])


def gt_reach(env, target, tlog, max_steps, maxnorm):
    """Ground-truth 'expert' reach: each step move the EE straight toward the target by up to
    ``maxnorm`` (the optimal action a perfect planner would pick under the same per-step action
    clip V-JEPA uses). This is the reference the V-JEPA rollout is compared against side-by-side."""
    for _ in range(max_steps):
        cur = env.get_ee_state()[:3]
        step = target - cur
        n = float(np.linalg.norm(step))
        if n <= 1e-3:
            break
        if n > maxnorm:
            step = step * (maxnorm / n)
        d = np.zeros(7)
        d[:3] = step
        env.apply_action(d)
        tlog.record(env, "ground_truth", d, None, float("nan"), target, float("nan"),
                    float(np.linalg.norm(env.get_ee_state()[:3] - target)))


def _gt_grasp(env, tlog):
    """Ground-truth scripted-expert grasp+lift (uses privileged cube xy): align above -> descend ->
    close -> lift -> settle. The 'how it should look' reference for the grasp_lift demo."""
    c = env.object_position()
    scripted(env, tlog, "gt_align", [c[0], c[1], c[2] + 0.12])
    c = env.object_position()
    grasp = [c[0], c[1], c[2] + 0.005]
    scripted(env, tlog, "gt_descend", grasp)
    scripted(env, tlog, "gt_close", grasp, gripper=1.0)
    scripted(env, tlog, "gt_lift", None, dz=0.12)
    scripted(env, tlog, "gt_settle", None, settle=20)


def _gt_pick_place(env, tlog):
    """Ground-truth scripted-expert pick-and-place: grasp -> transport over the zone CENTER ->
    lower -> open -> settle. The reference the V-JEPA place / pick_place rollout is compared to."""
    c = env.object_position()
    scripted(env, tlog, "gt_align", [c[0], c[1], c[2] + 0.12])
    c = env.object_position()
    grasp = [c[0], c[1], c[2] + 0.005]
    scripted(env, tlog, "gt_descend", grasp)
    scripted(env, tlog, "gt_close", grasp, gripper=1.0)
    scripted(env, tlog, "gt_lift", None, dz=0.14)
    zone = env.zone_center()
    scripted(env, tlog, "gt_transport", [zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.14])
    scripted(env, tlog, "gt_lower", [zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.02])
    scripted(env, tlog, "gt_open", None, gripper=-1.0)
    scripted(env, tlog, "gt_settle", None, settle=20)


def run_demo(env, ctx, args, out_path):
    """Build a side-by-side GROUND-TRUTH vs V-JEPA comparison GIF for the chosen task. Both rollouts
    run on the SAME seeded scene (target / cube position) under the same env actuator limit, played
    in sync so the viewer can compare V-JEPA's planned motion against the expert reference.

    reach       : GT = optimal straight-line reach; ours = V-JEPA closed-loop.
    grasp_lift  : GT = scripted-expert grasp+lift; ours = V-JEPA reaches the grasp pose (+scripted
                  close/lift).
    place       : GT = scripted-expert pick-and-place; ours = scripted grasp + V-JEPA place.
    pick_place  : GT = scripted-expert pick-and-place; ours = full V-JEPA composite (4/10/4)."""
    task = args.demo
    if task == "reach":
        target = _reach_target(env, ctx["rng"])
        env.reset()
        gt = TrialLogger(ctx["csv"], "demo_reach_gt", 0, ctx["fovy"], False)
        gt_reach(env, target, gt, args.reach_steps + 3, args.maxnorm)
        env.reset()
        ours = TrialLogger(ctx["csv"], "demo_reach_vjepa", 0, ctx["fovy"], False)

        def reach_goal(e, ctx):
            return e.capture_goal_image(pos=target, euler=EE_DOWN, camera="planning"), target

        run_vjepa_stages(env, ctx, ours, [Stage("vjepa_reach", reach_goal, args.reach_steps)], args)
        side_by_side_gif(gt.steps, ours.steps, out_path, "GROUND TRUTH (optimal)", "V-JEPA (ours)")
        return {"scene": target.tolist(),
                "gt_final_dist": float(gt.steps[-1]["stats_dist"]),
                "vjepa_final_dist": float(ours.steps[-1]["stats_dist"])}

    # object tasks: fix the scene so GT and V-JEPA share the same cube position
    cube_xy = _rand_cube_xy(ctx["rng"])
    ctx["fixed_cube_xy"] = cube_xy
    env.reset(cube_xy=cube_xy)
    gt = TrialLogger(ctx["csv"], f"demo_{task}_gt", 0, ctx["fovy"], True)
    (_gt_grasp if task == "grasp_lift" else _gt_pick_place)(env, gt)
    ours = TrialLogger(ctx["csv"], f"demo_{task}_vjepa", 0, ctx["fovy"], True)
    rec = TASKS[task](env, ctx, ours, args)
    side_by_side_gif(gt.steps, ours.steps, out_path,
                     "GROUND TRUTH (scripted expert)", "V-JEPA (ours)")
    return {"scene": [round(float(x), 3) for x in cube_xy],
            "vjepa_error": round(float(rec["error"]), 4), "vjepa_failure": rec["failure"] or "-",
            "vjepa_gates": rec["gates"]}


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
    """V-JEPA must REACH the grasp pose; only close + lift are scripted -- no privileged
    re-centering -- so the metrics reflect V-JEPA's own grasp-positioning. Precision error =
    ||object_xy - EE_xy|| BEFORE the close.

    single_goal:  one grasp sub-goal (goal = arm around the cube, gripper open).
    multistage:   pregrasp sub-goal (arm hovering above the cube) -> grasp sub-goal (arm around
                  the cube) -- the paper's long-horizon staged approach for pick.
    In both, the goal images leave the cube UNCHANGED (only the arm moves), so no held-object
    render is needed for the grasp reach."""
    cube_xy = ctx.get("fixed_cube_xy") or _rand_cube_xy(ctx["rng"])
    env.reset(cube_xy=cube_xy)
    c = env.object_position()
    grasp_pos = np.array([c[0], c[1], c[2] + 0.005])    # at the cube, fingers around it, open
    pregrasp_pos = np.array([c[0], c[1], c[2] + 0.10])  # hovering above the cube

    def grasp_goal(e, ctx):
        return e.capture_goal_image(pos=grasp_pos, euler=EE_DOWN, gripper=0.0,
                                    camera="planning"), grasp_pos

    def pregrasp_goal(e, ctx):
        return e.capture_goal_image(pos=pregrasp_pos, euler=EE_DOWN, gripper=0.0,
                                    camera="planning"), pregrasp_pos

    if args.protocol == "multistage":
        stages = [Stage("vjepa_pregrasp", pregrasp_goal, args.pregrasp_steps),
                  Stage("vjepa_grasp", grasp_goal, args.grasp_steps)]
    else:
        stages = [Stage("vjepa_grasp", grasp_goal, args.grasp_steps)]
    run_vjepa_stages(env, ctx, tlog, stages, args)

    # precision error: how well V-JEPA positioned the gripper over the object BEFORE the close
    grasp_xy_error = float(np.linalg.norm(env.object_position()[:2] - env.get_ee_state()[:2]))
    # ONLY scripted from here: close (holding the pose V-JEPA reached) then lift -- no re-centering.
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
    horizontal accuracy.

    single_goal:  one place sub-goal (held cube hovering over the zone).
    multistage:   place-vicinity sub-goal (held cube high over the zone) -> place-final sub-goal
                  (held cube lowered onto the zone) -- the paper's vicinity-then-place schedule.
    Both sub-goals render the CUBE carried in the gripper (held_object=True); the release lower +
    open is scripted. Note: the final sub-goal keeps the cube HELD (low over the zone) rather than
    showing a gripper-away 'resting' goal, because V-JEPA controls the held EE and cannot drive the
    gripper away without dragging the cube -- a held-low goal is the reachable placement target."""
    cube_xy = ctx.get("fixed_cube_xy") or _rand_cube_xy(ctx["rng"])
    env.reset(cube_xy=cube_xy)
    _scripted_grasp(env, tlog)  # reliable grasp to isolate the PLACEMENT skill
    if not env.gripper_holds_object():
        err = float(np.linalg.norm(env.object_position()[:2] - env.zone_center()))
        tlog.record(env, "final", None, None, float("nan"), None, float("nan"), err,
                    success=0, failure="grasp_failed_pre_place")
        return {"error": err, "gates": {g: False for g in GATE_SPEC["place"]},
                "failure": "grasp_failed_pre_place", "metrics": {}}
    zone = env.zone_center()
    hover_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.12])  # vicinity (high)
    low_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.04])    # final (lowered)

    def hover_goal(e, ctx):
        return e.capture_goal_image(pos=hover_pos, euler=EE_DOWN, gripper=1.0,
                                    camera="planning", held_object=True), hover_pos

    def low_goal(e, ctx):
        return e.capture_goal_image(pos=low_pos, euler=EE_DOWN, gripper=1.0,
                                    camera="planning", held_object=True), low_pos

    if args.protocol == "multistage":
        stages = [Stage("vjepa_place_vicinity", hover_goal, args.vicinity_steps),
                  Stage("vjepa_place_final", low_goal, args.place_steps)]
    else:
        stages = [Stage("vjepa_place", hover_goal, args.place_steps)]
    run_vjepa_stages(env, ctx, tlog, stages, args)

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
    tlog.record(env, "final", None, None, float("nan"), low_pos, float("nan"), place_xy_error,
                success=int(res.success), failure=res.failure_type or "")
    return {"error": place_xy_error, "gates": gates, "failure": res.failure_type or "",
            "metrics": res.metrics}


def task_pick_place(env, ctx, tlog, args):
    """Paper-faithful composite Pick-and-Place (arXiv 2506.09985 Sec. 4.2): V-JEPA does the WHOLE
    task -- grasp the object, transport it, and place it -- via THREE sub-goal images on the paper's
    fixed 4 / 10 / 4 time-step schedule (sub-goals switch on a step budget, not on reaching):

      1. grasp sub-goal    (4 steps): goal = the arm at the cube, grasp-ready (cube unchanged).
      2. vicinity sub-goal (10 steps): goal = the held cube in the vicinity of the zone (high hover).
      3. place sub-goal     (4 steps): goal = the held cube at the zone (lowered).

    Only the gripper finger actuation is scripted (close after the grasp sub-goal, open after the
    place sub-goal); all end-effector motion -- including the grasp reach -- is planned by V-JEPA, so
    this tests the compose-atomic-skills ability the paper measures. The paper drives the gripper via
    the CEM ``close_gripper`` schedule; we script the close at the grasp->transport transition
    instead, consistent with our honest V-JEPA-does-spatial / scripted-does-gripper decomposition.
    Precision error = ||object_xy_final - zone_xy||. A grasp miss fails the whole task (honest
    composite)."""
    cube_xy = ctx.get("fixed_cube_xy") or _rand_cube_xy(ctx["rng"])
    env.reset(cube_xy=cube_xy)
    c = env.object_position()
    grasp_pos = np.array([c[0], c[1], c[2] + 0.005])  # at the cube, fingers around it, open

    def grasp_goal(e, ctx):
        return e.capture_goal_image(pos=grasp_pos, euler=EE_DOWN, gripper=0.0,
                                    camera="planning"), grasp_pos

    # 1) grasp sub-goal (4 fixed steps): V-JEPA reaches the grasp pose, gripper frozen open
    run_vjepa_stages(env, ctx, tlog,
                     [Stage("vjepa_pnp_grasp", grasp_goal, args.pnp_grasp_steps, fixed_steps=True)],
                     args)
    grasp_xy_error = float(np.linalg.norm(env.object_position()[:2] - env.get_ee_state()[:2]))
    # scripted close (finger actuation only, holding the pose V-JEPA reached)
    scripted(env, tlog, "close", None, gripper=1.0)
    grasped = bool(env.gripper_holds_object())

    # 2+3) transport + place sub-goals: V-JEPA drives the held cube (goals render the held cube)
    zone = env.zone_center()
    vicinity_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.12])
    place_pos = np.array([zone[0], zone[1], TABLE_TOP_Z + CUBE_HALF + 0.04])

    def vicinity_goal(e, ctx):
        return e.capture_goal_image(pos=vicinity_pos, euler=EE_DOWN, gripper=1.0,
                                    camera="planning", held_object=True), vicinity_pos

    def place_goal(e, ctx):
        return e.capture_goal_image(pos=place_pos, euler=EE_DOWN, gripper=1.0,
                                    camera="planning", held_object=True), place_pos

    run_vjepa_stages(env, ctx, tlog, [
        Stage("vjepa_pnp_vicinity", vicinity_goal, args.pnp_vicinity_steps, fixed_steps=True),
        Stage("vjepa_pnp_place", place_goal, args.pnp_place_steps, fixed_steps=True),
    ], args)

    # scripted release: lower straight down at V-JEPA's reached xy, open, settle
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
        "grasped": grasped,
        "upright": float(np.degrees(env.object_tilt())) < spec["tilt_max_deg"],
        "stable": float(env.object_speed()) < spec["v_settle"],
        "released": bool(env.object_released()),
    }
    # a grasp miss fails the whole composite regardless of where the (un-held) cube ends up
    failure = res.failure_type or ""
    if not grasped:
        failure = "grasp_failed"
    tlog.record(env, "final", None, None, float("nan"), place_pos, float("nan"), place_xy_error,
                success=int(res.success and grasped), failure=failure)
    metrics = dict(res.metrics or {})
    metrics.update(grasp_xy_error=round(grasp_xy_error, 4), grasped=grasped)
    return {"error": place_xy_error, "gates": gates, "failure": failure, "metrics": metrics}


# ----------------------------------------------------------------------------- task registry
TASKS = {"reach": task_reach, "grasp_lift": task_grasp_lift, "place": task_place,
         "pick_place": task_pick_place}

# BUNDLE_TASKS + validate_task_mode are imported from src.bench.thresholds (pure + unit-tested);
# LEGACY_TASKS mirrors this scripted-function registry (guard against drift).
assert set(TASKS) == set(LEGACY_TASKS)


# Candidate planning cameras for the camera-salience experiment (partial overrides of
# PLANNING_CAMERA). A_current == the validated default; B keeps the angle and moves closer (more
# object/gripper pixels, same action frame); C is a DROID-like left-exo view (angle change ->
# action-frame confound, so log action alignment separately). See scripts/camera_salience_probe.py.
CAMERA_PRESETS = {
    "A_current":   {},
    "B_closer":    {"distance": 1.05},
    "C_droidlike": {"azimuth": -135.0, "elevation": -30.0, "distance": 1.3},
}

# image key -> saved qpos key, for re-rendering goal images from a candidate camera (so the
# observation and the goal image always share the same viewpoint).
_GOAL_QPOS = {"goal": "qpos_goal", "goal_1": "qpos_goal_1", "goal_2": "qpos_goal_2"}


def _rerender_goals(env, arr, img_keys):
    """Re-render each goal image from its saved qpos using the env's current planning camera, so a
    camera override applies to goals as well as observations. Returns {img_key: rgb}. Leaves the env
    state disturbed -- the caller restores the trial start (set_state(qpos0)) afterward."""
    out = {}
    for k in img_keys:
        qk = _GOAL_QPOS.get(k)
        if qk is not None and qk in arr:
            env.set_state(arr[qk])
            out[k] = env.render(camera="planning")
    return out


def _fixed_goal(img, target):
    """A Stage goal_fn that returns a SAVED goal image + its target xyz (for the bundle benchmark),
    instead of capturing a goal at runtime."""
    tgt = np.asarray(target, dtype=float).reshape(3)
    img = np.ascontiguousarray(img[..., :3])  # drop any alpha; match env.render (HxWx3 uint8)

    def goal_fn(env, ctx):
        return img, tgt
    return goal_fn


def run_bundle_trial(env, ctx, tlog, args, bundle):
    """Run one FIXED-BUNDLE trial: restore the saved start state, plan with V-JEPA to the SAVED goal
    images (auto-switching sub-goals on the step budget, like the released V-JEPA 2-AC pick-and-place
    schedule), script only the gripper at stage transitions, and score from hidden privileged state.
    Returns {error, gates, failure, metrics}."""
    meta, arr, imgs = bundle.meta, bundle.arrays, bundle.images
    task = meta["task"]
    start_grasped = bool(meta.get("start_grasped", False))

    # restore the exact recorded scenario (start pose + object + zone)
    env.reset()
    zone_xy = arr.get("zone")
    if zone_xy is not None and np.all(np.isfinite(zone_xy)):
        env.set_zone_xy(float(zone_xy[0]), float(zone_xy[1]))
    if getattr(args, "planning_camera", None):
        # camera override active -> re-render goals from saved qpos so obs and goals share the view
        imgs = _rerender_goals(env, arr, list(imgs.keys()))
    env.set_state(arr["qpos0"], gripper=(1.0 if start_grasped else 0.0),
                  settle=(8 if start_grasped else 0))

    def stg(name, img_key, tgt_key, steps, fixed=False):
        return Stage(name, _fixed_goal(imgs[img_key], arr[tgt_key]), steps, fixed_steps=fixed)

    if task == "grasp":
        # V-JEPA reaches the grasp pose (goal = object just grabbed); close + lift are scripted.
        run_vjepa_stages(env, ctx, tlog, [stg("vjepa_grasp", "goal", "grasp_pos", args.grasp_steps)], args)
        error = float(np.linalg.norm(env.object_position()[:2] - env.get_ee_state()[:2]))
        obj_z0 = float(env.object_position()[2])
        scripted(env, tlog, "close", None, gripper=1.0)
        scripted(env, tlog, "lift", None, dz=0.12)
        scripted(env, tlog, "settle", None, settle=20)
        obj = env.object_position()
        gates = {"lifted": float(obj[2] - obj_z0) > 0.04, "held": bool(env.gripper_holds_object()),
                 "upright": float(np.degrees(env.object_tilt())) < 30.0,
                 "stable": float(env.object_speed()) < 0.05}
        target = arr["grasp_pos"]

    elif task in ("reach_with_object", "grasp_and_reach"):
        if task == "grasp_and_reach":
            run_vjepa_stages(env, ctx, tlog,
                             [stg("vjepa_gnr_grasp", "goal_1", "grasp_pos", args.pnp_grasp_steps)], args)
            scripted(env, tlog, "close", None, gripper=1.0)
        run_vjepa_stages(env, ctx, tlog,
                         [stg("vjepa_reach_obj", "goal", "goal_ee", args.rwo_steps)], args)
        scripted(env, tlog, "settle", None, settle=10)
        obj = env.object_position()
        goal_obj = np.asarray(arr["goal_object"], dtype=float)
        error = float(np.linalg.norm(obj - goal_obj))
        held = bool(env.gripper_holds_object())
        gates = {"held": held, "upright": float(np.degrees(env.object_tilt())) < 30.0}
        target = arr["goal_ee"]

    else:  # pick_place -- 3 sub-goals on the fixed 4 / 10 / 4 schedule
        run_vjepa_stages(env, ctx, tlog,
                         [stg("vjepa_pnp_grasp", "goal_1", "grasp_pos", args.pnp_grasp_steps, True)], args)
        scripted(env, tlog, "close", None, gripper=1.0)
        grasped = bool(env.gripper_holds_object())
        run_vjepa_stages(env, ctx, tlog, [
            stg("vjepa_pnp_vicinity", "goal_2", "vicinity_pos", args.pnp_vicinity_steps, True),
            stg("vjepa_pnp_place", "goal", "place_pos", args.pnp_place_steps, True)], args)
        cur = env.get_ee_state()[:3]
        scripted(env, tlog, "lower", [cur[0], cur[1], TABLE_TOP_Z + 0.05])
        scripted(env, tlog, "open", None, gripper=-1.0)
        scripted(env, tlog, "settle", None, settle=30)
        obj = env.object_position()
        error = float(np.linalg.norm(obj[:2] - env.zone_center()))
        gates = {"grasped": grasped, "upright": float(np.degrees(env.object_tilt())) < 25.0,
                 "stable": float(env.object_speed()) < 0.05, "released": bool(env.object_placed())}
        target = arr["place_pos"]

    success, failure = bundle_classify(task, error, gates, THRESHOLDS[task])
    tlog.record(env, "final", None, None, float("nan"), target, float("nan"), error,
                success=int(success), failure=failure)
    return {"error": error, "gates": gates, "failure": failure, "metrics": {}}



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


# ----------------------------------------------------------------------------- bundle benchmark
def run_bundle_benchmark(args, run_id, config, encoder, plan, encode_goal_factory,
                         csv, trial_csv, run_dir, viz_dir, report_dir,
                         device, dev, tokens_per_frame, autocast_dtype):
    """Run the fixed-bundle benchmark: for each (task, object) load the saved bundles from
    ``args.bundles/<task>/<object>/``, restore each recorded scenario, plan with V-JEPA to the saved
    goal images, and score from hidden privileged state. Reports per (task, object)."""
    import torch

    bundles_root = (args.bundles if os.path.isabs(args.bundles)
                    else os.path.join(_REPO_ROOT, args.bundles))
    if not os.path.isdir(bundles_root):
        raise SystemExit(f"--bundles dir not found: {bundles_root} (generate with "
                         f"scripts/generate_task_bundles.py, or pass the correct path)")
    summaries, all_records = [], {}
    total_loaded = 0
    for task in args.tasks:
        for obj in args.objects:
            obj_dir = os.path.join(bundles_root, task, obj)
            if not os.path.isdir(obj_dir):
                logger.warning("no bundles for %s/%s under %s -- skipping", task, obj, bundles_root)
                continue
            ids = sorted(d for d in os.listdir(obj_dir)
                         if os.path.isdir(os.path.join(obj_dir, d)))
            if not ids:
                logger.warning("%s/%s has 0 bundles -- skipping", task, obj)
                continue
            if len(ids) < args.trials:
                logger.warning("%s/%s has only %d bundles < requested --trials %d; running all %d",
                               task, obj, len(ids), args.trials, len(ids))
            ids = ids[: args.trials]
            total_loaded += len(ids)
            cam_override = CAMERA_PRESETS.get(args.planning_camera) if args.planning_camera else None
            env = FrankaDroidEnv(render_width=CROP, render_height=CROP, max_translation=0.13,
                                 add_object=True, add_zone=True, object_type=obj, add_distractors=True,
                                 planning_camera=cam_override)
            fovy = float(env.model.vis.global_.fovy)
            ctx = {"encoder": encoder, "plan": plan, "encode_goal": encode_goal_factory(),
                   "cem_kw": dict(encode_fn=encode, device=device, tokens_per_frame=tokens_per_frame,
                                  dev=dev, autocast_dtype=autocast_dtype,
                                  freeze_gripper=not args.plan_gripper)}
            label = f"{task}/{obj}"
            pair_off = zlib.crc32(label.encode()) & 0xFFFF  # per-(task,object) seed offset
            records, tlogs = [], []
            for ti, bid in enumerate(ids):
                bundle = TaskBundle.load(os.path.join(obj_dir, bid))
                torch.manual_seed(args.seed * 100003 + pair_off * 997 + ti)  # CEM sampling (bf16 noisy)
                tlog = TrialLogger(csv, label, ti, fovy, True)
                tlog.record_qpos = bool(args.replay_record)
                t0 = time.perf_counter()
                rec = run_bundle_trial(env, ctx, tlog, args, bundle)
                dt = time.perf_counter() - t0
                if args.replay_record:
                    os.makedirs(args.replay_record, exist_ok=True)
                    tlog.save_rollout(os.path.join(args.replay_record, f"{task}_{obj}_{bid}.npz"), obj)
                vjepa_rows = [r for r in tlog.rows if str(r["phase"]).startswith("vjepa")]
                energies = [_pf(r["energy"]) for r in vjepa_rows if _pf(r["energy"]) is not None]
                cems = [_pf(r["cem_s"]) for r in tlog.rows if _pf(r["cem_s"]) is not None]
                thr_flags = {t: success_at(rec["error"], rec["gates"], t) for t in THRESHOLDS[task]}
                rec.update(task=task, label=label, trial=ti, bundle_id=bid, wall_s=round(dt, 1),
                           final_energy=(energies[-1] if energies else None),
                           vjepa_steps=len(vjepa_rows), total_steps=len(tlog.rows),
                           mean_cem_s=(float(np.mean(cems)) if cems else float("nan")),
                           thr_flags=thr_flags, all_success=any(thr_flags.values()))
                records.append(rec)
                tlogs.append(tlog)
                trial_csv.log(label, ti, bid, round(rec["error"], 4), int(rec["all_success"]),
                              rec["failure"], _r(rec["final_energy"]), rec["vjepa_steps"],
                              rec["total_steps"], _r(rec["mean_cem_s"]), rec["wall_s"],
                              json.dumps({str(k): int(v) for k, v in thr_flags.items()}))
                logger.info("[%s t%02d %s] err=%.4f gates=%s fail=%s | vjepa=%d steps | %.0fs",
                            label, ti, bid, rec["error"], rec["gates"], rec["failure"] or "-",
                            rec["vjepa_steps"], dt)
            env.close()

            for i in select_for_viz(records):
                frames = tlogs[i].build_frames()
                gif_name = f"{task}_{obj}_t{i}.gif"
                contact_name = f"{task}_{obj}_t{i}_contact.png"
                imageio.mimsave(os.path.join(viz_dir, gif_name), frames, fps=2, loop=0)
                save_contact_sheet(frames, os.path.join(viz_dir, contact_name),
                                   phases=tlogs[i].phases)
                for name in (gif_name, contact_name):  # copy selected viz into the committed report
                    shutil.copy(os.path.join(viz_dir, name), os.path.join(report_dir, name))
            write_task_plots(records, task, os.path.join(report_dir, f"{task}_{obj}_summary.png"))
            s = summarize_task(records, task)
            s["label"] = label
            summaries.append(s)
            all_records[label] = records

    if not summaries:
        raise SystemExit(f"no bundles matched tasks={args.tasks} objects={args.objects} under "
                         f"{bundles_root}; nothing to run (check the path, task/object names).")
    logger.info("loaded %d bundles across %d (task, object) pairs", total_loaded, len(summaries))
    _write_bundle_summary(report_dir, run_id, config, summaries, args, bundles_root)
    logger.info("-" * 64)
    logger.info("FIXED-BUNDLE PRECISION SUMMARY (samples=%d T=%d cem_steps=%d maxnorm=%.3f)",
                args.samples, args.rollout, args.cem_steps, args.maxnorm)
    for s in summaries:
        thr_str = " ".join(f"@{_cm(t)}cm={s[f'succ@{t}']*100:.0f}%" for t in THRESHOLDS[s["task"]])
        logger.info("  %-24s n=%d mean=%.3f median=%.3f | %s",
                    s["label"], s["n"], s["mean_err"], s["median_err"], thr_str)
    logger.info("run log (gitignored): %s", os.path.relpath(run_dir, _REPO_ROOT))
    logger.info("committed report: %s", os.path.relpath(report_dir, _REPO_ROOT))
    logger.info("done")


def _write_bundle_summary(report_dir, run_id, config, summaries, args, bundles_root):
    """Write summary.csv + summary.md for the fixed-bundle run (rows keyed by task/object)."""
    scsv = clog.CSVLogger(
        os.path.join(report_dir, "summary.csv"),
        "task_object", "n", "mean_err_m", "median_err_m", "p90_err_m",
        "mean_vjepa_steps", "mean_cem_s", "thresholds_m", "success_rates",
    )
    for s in summaries:
        rates = {f"{t}": round(s[f"succ@{t}"], 3) for t in THRESHOLDS[s["task"]]}
        scsv.log(s["label"], s["n"], round(s["mean_err"], 4), round(s["median_err"], 4),
                 round(s["p90_err"], 4), round(s["mean_vjepa_steps"], 1), round(s["mean_cem_s"], 2),
                 str(THRESHOLDS[s["task"]]), json.dumps(rates))

    lines = [
        f"# Fixed-bundle closed-loop benchmark -- run {run_id}",
        "",
        f"Config: model **{config['model']}**, samples **{args.samples}**, cem_steps "
        f"**{args.cem_steps}**, rollout **T={args.rollout}**, topk **{args.topk}**, maxnorm "
        f"**{args.maxnorm} m**, momentum 0.15/0.15, dtype **{args.dtype}**. Fixed saved bundles from "
        f"`{os.path.relpath(bundles_root, _REPO_ROOT)}` ({args.trials} trials/(task,object)). "
        f"Commit `{config['git_commit'][:10]}`.",
        "",
        "Success = Euclidean delta within the swept sphere radius `x` AND the task's physical gates "
        "(grasp: grasped+lifted+upright+stable; reach_with_object/grasp_and_reach: held+upright; "
        "pick_place: grasped+released+upright+stable), judged only from hidden MuJoCo truth. For "
        "pick_place, success cares only that the object lands in the zone and is released (arm "
        "distance is ignored).",
        "",
        "| task / object | n | mean Δ (cm) | median (cm) | p90 (cm) | success @ x | main failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        succ = " ".join(f"@{_cm(t)}cm **{s[f'succ@{t}']*100:.0f}%**" for t in THRESHOLDS[s["task"]])
        fails = ", ".join(f"{k} x{v}" for k, v in sorted(s["failures"].items(),
                                                         key=lambda kv: -kv[1])) or "-"
        lines.append(f"| **{s['label']}** | {s['n']} | {s['mean_err']*100:.1f} | "
                     f"{s['median_err']*100:.1f} | {s['p90_err']*100:.1f} | {succ} | {fails} |")
    lines += ["", f"Per-(task,object) figures: `{{task}}_{{object}}_summary.png`. Full per-step log: "
              f"`{os.path.relpath(os.path.join(_REPO_ROOT, 'logs', 'closed_loop_runs', run_id), _REPO_ROOT)}`."]
    with open(os.path.join(report_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- main
def main() -> None:
    import torch

    from utils.mpc_utils import cem, compute_new_pose

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=["reach", "grasp_lift", "place"],
                   choices=list(TASKS) + [t for t in BUNDLE_TASKS if t not in TASKS])
    p.add_argument("--bundles", default=None,
                   help="run on FIXED saved task bundles from this dir (tasks/<task>/<object>/...) "
                        "instead of randomizing per trial; enables the fixed-bundle benchmark")
    p.add_argument("--objects", nargs="+", default=["cup", "box"], choices=["cup", "box"],
                   help="objects to run in --bundles mode (each task runs per object)")
    p.add_argument("--rwo-steps", type=int, default=10,
                   help="CEM budget for the held-object reach stage (reach_with_object / "
                        "grasp_and_reach), a longer paper-like traverse")
    p.add_argument("--trials", type=int, default=1, help="trials per task (smoke: 1-5; full: 50)")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--chunk", type=int, default=400,
                   help="predictor sub-batch size over CEM samples (0/>=samples = whole batch); "
                        "caps peak VRAM for samples=800 (mathematically identical, samples are "
                        "independent). Default 400: 200/400 run whole-batch, 800 splits in two")
    p.add_argument("--cem-steps", type=int, default=10)
    p.add_argument("--rollout", type=int, default=1,
                   help="CEM planning horizon T (paper p.37 = 1; each candidate is one next action)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--maxnorm", type=float, default=0.05, help="CEM per-axis action clip (m)")
    p.add_argument("--plan-gripper", action="store_true", default=False,
                   help="let CEM plan the gripper action (paper-faithful: goal images show the "
                        "closed/open gripper, so the planner can reach them). Default off = gripper "
                        "frozen during the V-JEPA reach and scripted afterward (honest-separation).")
    p.add_argument("--replay-record", default=None,
                   help="dir to save per-trial qpos rollouts (npz) for the interactive 3D replay "
                        "viewer (scripts/replay_rollout_viewer.py). Bundle mode only.")
    p.add_argument("--planning-camera", choices=list(CAMERA_PRESETS), default=None,
                   help="override the planning camera (bundle mode): re-renders goal images from the "
                        "same view so obs and goals stay consistent. A_current=validated default, "
                        "B_closer=same angle tighter, C_droidlike=left-exo (action-frame confound).")
    p.add_argument("--pos-tol", type=float, default=0.015,
                   help="early-stop the CEM loop when EE is within this of the goal (m); set to the "
                        "tightest precision threshold so the loop does not quit before the strictest "
                        "success cutoff can be measured")
    p.add_argument("--protocol", choices=["single_goal", "multistage"], default="single_goal",
                   help="single_goal: one goal image per task (baseline). multistage: paper-like "
                        "sub-goal schedule (grasp: pregrasp->grasp; place: vicinity->final)")
    p.add_argument("--reach-steps", type=int, default=5)
    p.add_argument("--pregrasp-steps", type=int, default=3,
                   help="multistage grasp: CEM budget for the pregrasp (hover) sub-goal")
    p.add_argument("--grasp-steps", type=int, default=6)
    p.add_argument("--vicinity-steps", type=int, default=6,
                   help="multistage place: CEM budget for the vicinity (high hover) sub-goal")
    p.add_argument("--place-steps", type=int, default=8)
    # Paper-faithful pick_place schedule (arXiv 2506.09985 Sec. 4.2): fixed 4 / 10 / 4 time-steps.
    p.add_argument("--pnp-grasp-steps", type=int, default=4,
                   help="pick_place grasp sub-goal budget (paper: 4 fixed steps)")
    p.add_argument("--pnp-vicinity-steps", type=int, default=10,
                   help="pick_place vicinity sub-goal budget (paper: 10 fixed steps)")
    p.add_argument("--pnp-place-steps", type=int, default=4,
                   help="pick_place final-place sub-goal budget (paper: 4 fixed steps)")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0, help="seed for per-trial randomized init")
    p.add_argument("--tag", default="full", help="report subdir: results/benchmarks/closed_loop_<tag>")
    p.add_argument("--demo", choices=["reach", "grasp_lift", "place", "pick_place"], default=None,
                   help="LEGACY (random-task) demo: build a side-by-side ground-truth vs V-JEPA "
                        "comparison GIF for the given legacy task and exit. Does NOT use the fixed "
                        "bundles; the fixed-bundle tasks are not available here.")
    p.add_argument("--viz-only-selected", action="store_true", default=True,
                   help="save GIFs only for ~3 best/median/worst trials (default on)")
    args = p.parse_args()

    # In --bundles mode, default to the fixed-bundle task set unless the user asked for specific tasks.
    if args.bundles and args.tasks == ["reach", "grasp_lift", "place"]:
        args.tasks = list(BUNDLE_TASKS)

    # Guard task/mode combinations: fixed-bundle-only tasks (grasp / reach_with_object /
    # grasp_and_reach) need saved bundles; the legacy random tasks (reach / grasp_lift / place) have
    # no bundles. ``pick_place`` exists in BOTH registries and is allowed either way.
    _mode_err = validate_task_mode(args.tasks, args.bundles)
    if _mode_err:
        raise SystemExit(_mode_err)

    # Stop-guard: if this flag file exists, exit immediately (before loading the model). Lets a
    # detached multi-run launcher be halted cleanly -- already-running invocations finish from their
    # loaded bytecode, while the NEXT fresh invocation reads this and no-ops. Remove the file to resume.
    _stop_flag = os.path.join(_REPO_ROOT, "logs", "full_bench", "STOP")
    if os.path.exists(_stop_flag):
        logger.warning("STOP flag present (%s) -- exiting without running.", _stop_flag)
        return

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
                "topk": args.topk, "maxnorm": args.maxnorm, "chunk": args.chunk,
                "momentum_mean": 0.15, "momentum_std": 0.15,
                "objective": "mean-L1 in layer-norm'd latent, gripper axis frozen"},
        "tasks": args.tasks, "trials_per_task": args.trials, "seed": args.seed,
        "protocol": args.protocol,
        "pos_tol": args.pos_tol,
        "max_vjepa_steps": {"reach": args.reach_steps, "pregrasp": args.pregrasp_steps,
                            "grasp": args.grasp_steps, "place_vicinity": args.vicinity_steps,
                            "place": args.place_steps, "reach_with_object": args.rwo_steps,
                            "pnp_4_10_4": [args.pnp_grasp_steps, args.pnp_vicinity_steps,
                                           args.pnp_place_steps]},
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
    if args.bundles:
        # Fixed-bundle mode: the start pose, object type, and zone come from the saved bundle, so the
        # cube-specific/random-start fields above do NOT describe this run. Record the true provenance.
        config["mode"] = "fixed_bundles"
        config["bundles"] = {
            "dir": args.bundles, "objects": list(args.objects), "tasks": list(args.tasks),
            "trials_per_task_object": args.trials, "rwo_steps": args.rwo_steps,
            "object_rest_half_z_m": {o: OBJECT_SPECS[o]["rest_half_z"] for o in args.objects},
            "planning_camera": args.planning_camera,
            "planning_camera_override": (CAMERA_PRESETS.get(args.planning_camera)
                                         if args.planning_camera else None),
            "plan_gripper": bool(args.plan_gripper),
        }
        config["env"] = {
            "embodiment": "Franka Panda + Robotiq 2F-85 (FrankaDroidEnv)",
            "camera": "PLANNING_CAMERA (az45_el45 free cam)", "crop": CROP,
            "objects": list(args.objects), "add_distractors": True,
            "start": "restored from saved bundle qpos0 (fixed, not randomized)",
            "place_zone": "restored per bundle (mocap place zone)",
        }
    else:
        config["mode"] = "random_per_trial"
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 64)
    logger.info("closed-loop benchmark | protocol=%s tasks=%s trials=%d | samples=%d cem_steps=%d "
                "T=%d maxnorm=%.3f", args.protocol, args.tasks, args.trials, args.samples,
                args.cem_steps, args.rollout, args.maxnorm)
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

    if args.demo:
        demo_dir = os.path.join(_REPO_ROOT, "results", "benchmarks", "closed_loop_smoke")
        os.makedirs(demo_dir, exist_ok=True)
        needs_obj = args.demo in ("grasp_lift", "place", "pick_place")
        env = FrankaDroidEnv(render_width=CROP, render_height=CROP, max_translation=0.13,
                             add_object=needs_obj, add_zone=needs_obj)
        seed = args.seed * 100003 + TASK_SEED_OFFSET[args.demo]
        ctx = {"encoder": encoder, "plan": plan, "encode_goal": encode_goal_factory(),
               "csv": csv, "fovy": float(env.model.vis.global_.fovy),
               "rng": np.random.default_rng(seed),
               "cem_kw": dict(encode_fn=encode, device=device, tokens_per_frame=tokens_per_frame,
                              dev=dev, autocast_dtype=autocast_dtype,
                              freeze_gripper=not args.plan_gripper)}
        out = os.path.join(demo_dir, f"demo_{args.demo}_compare.gif")
        torch.manual_seed(seed)
        info = run_demo(env, ctx, args, out)
        env.close()
        logger.info("demo GIF -> %s | %s", os.path.relpath(out, _REPO_ROOT), json.dumps(info))
        return

    trial_csv = clog.CSVLogger(
        os.path.join(run_dir, "trials.csv"),
        "task", "trial", "bundle_id", "error_m", "success_loose", "failure", "final_energy",
        "vjepa_steps", "total_steps", "mean_cem_s", "wall_s", "success_at_thresholds",
    )

    if args.bundles:
        run_bundle_benchmark(args, run_id, config, encoder, plan, encode_goal_factory,
                             csv, trial_csv, run_dir, viz_dir, report_dir,
                             device, dev, tokens_per_frame, autocast_dtype)
        return

    all_records = {}
    for task in args.tasks:
        needs_obj = task in ("grasp_lift", "place", "pick_place")
        env = FrankaDroidEnv(render_width=CROP, render_height=CROP, max_translation=0.13,
                             add_object=needs_obj, add_zone=needs_obj)
        fovy = float(env.model.vis.global_.fovy)
        task_off = TASK_SEED_OFFSET[task] * 100003
        ctx = {"encoder": encoder, "plan": plan, "encode_goal": encode_goal_factory(),
               "cem_kw": dict(encode_fn=encode, device=device, tokens_per_frame=tokens_per_frame,
                              dev=dev, autocast_dtype=autocast_dtype,
                              freeze_gripper=not args.plan_gripper)}
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
            trial_csv.log(task, trial, "", round(rec["error"], 4), int(rec["all_success"]),
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
        f"Config: model **{config['model']}**, protocol **{args.protocol}**, samples "
        f"**{args.samples}**, cem_steps **{args.cem_steps}**, rollout **T={args.rollout}**, topk "
        f"**{args.topk}**, maxnorm **{args.maxnorm} m**, dtype **{args.dtype}**, "
        f"**{args.trials} trials/task**, seed {args.seed}.",
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
