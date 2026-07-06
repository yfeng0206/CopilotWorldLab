"""Draw an annotated flowchart of the closed-loop V-JEPA 2-AC benchmark: the nested loop structure
(benchmark -> trial -> task sub-goal schedule -> MPC time-step -> CEM optimization) with every
hyperparameter labeled at the level where it applies. Output: a committed PNG for the docs.
Pure matplotlib (no model / no env), safe to run alongside a benchmark.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "results/benchmarks/closed_loop_smoke/benchmark_flowchart.png"

fig, ax = plt.subplots(figsize=(15, 10.5), dpi=130)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, lines, fc, ec="#333", tc="#111", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top", fontsize=fs + 1.5,
            fontweight="bold", color=tc)
    ax.text(x + 2.2, y + h - 6.2, "\n".join(lines), ha="left", va="top", fontsize=fs,
            color=tc, family="monospace")

def arrow(x1, y1, x2, y2, txt=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.8, color="#555"))
    if txt:
        ax.text((x1 + x2) / 2 + 1.5, (y1 + y2) / 2, txt, fontsize=8.2, color="#444",
                style="italic", ha="left", va="center")

ax.text(50, 98.5, "Closed-loop V-JEPA 2-AC benchmark  --  how one test runs (and every hyperparameter)",
        ha="center", va="top", fontsize=14, fontweight="bold")

# 1 BENCHMARK
box(2, 84, 46, 12, "1. BENCHMARK  (per task)", [
    "trials = 50 (paper: 10; our retry: 75)   -- randomized episodes",
    "seed = 0   -- fixes cube/target + CEM sampling (reproducible)",
    "tasks: reach | grasp_lift | pick_place",
], "#e8f0fe")
# 2 TRIAL
box(2, 68, 46, 13.5, "2. TRIAL  (one episode)", [
    "reset env: randomize cube xy + target (seeded)",
    "build GOAL IMAGE(s) -- what 'done' looks like:",
    "  reach/grasp = 1 goal   pick_place = 3 goals (4/10/4)",
    "obs = 256x256 RGB (PLANNING_CAMERA) + 7-D EE state",
], "#e6f4ea")
# 3 TASK SCHEDULE
box(2, 47, 46, 18.5, "3. TASK = sequence of STAGES (sub-goals)", [
    "reach       : [reach x5]              (1 goal)",
    "grasp_lift  : [grasp x6] + script close/lift   (1 goal)",
    "pick_place  : [grasp x4]->[vicinity x10]->[place x4]",
    "              = the paper's 4 / 10 / 4 schedule (3 goals)",
    "",
    "each number = # of MPC time-steps toward that sub-goal;",
    "gripper open/close/lift/open done by SCRIPT at transitions",
], "#fef7e0")

arrow(25, 84, 25, 81.5)
arrow(25, 68, 25, 65.5)

# 4 MPC TIME-STEP (middle column)
box(52, 60, 46, 22, "4. MPC TIME-STEP  (one observe->plan->act cycle)", [
    "repeat N times (the 4/10/4 / per-stage counts):",
    "  a. render image + read EE state",
    "  b. ENCODE image -> latent  (ViT-g, 256 tokens/frame)",
    "  c. CEM optimize the next action  (box 5)  ",
    "  d. EXECUTE only the 1st action  (receding horizon)",
    "     - gripper axis FROZEN (script controls gripper)",
    "     - EE move clipped to env max_translation = 0.13 m",
    "  e. re-observe, repeat  |  early-stop if <pos_tol (reach/grasp)",
], "#fde7e9")
arrow(48, 56, 52, 66, "for each stage")

# 5 CEM optimization (right, detailed)
box(52, 30, 46, 27, "5. CEM  (Cross-Entropy Method -- picks the action)", [
    "repeat cem_steps = 10 iterations:",
    "  1. sample  samples = 200/400/800  action-trajectories",
    "        each trajectory has length  rollout T = 2  (2 steps ahead)",
    "        action = [dx,dy,dz, d-gripper]; per-axis clip |a| <= maxnorm",
    "        maxnorm = 0.05 m  (paper-text 0.075 -> our retry)",
    "  2. WORLD MODEL predicts the T future latents for each",
    "        (predictor chunked in sub-batches of  chunk=400  for VRAM)",
    "  3. score = mean-L1 distance (predicted final latent vs GOAL latent)",
    "  4. keep  topk = 10  best; update sampling mean/std",
    "        momentum_mean = momentum_std = 0.15  (smoothing)",
    "  -> return the mean action of the final distribution",
], "#f3e8fd", fs=8.6)
arrow(75, 60, 75, 57)

# 6 SUCCESS
box(2, 27, 46, 15, "6. SUCCESS  (hidden MuJoCo truth -- model never sees it)", [
    "record one continuous ERROR per trial:",
    "  reach: ||EE-target||  grasp: ||obj_xy-EE_xy||",
    "  pick_place: ||obj_xy_final - zone_xy||",
    "success@t = (error < t) AND physical gates",
    "  gates: grasped / lifted / held / upright / stable / released",
    "thresholds give a PRECISION CURVE (not one cutoff)",
], "#e2f0f0")
arrow(25, 47, 25, 42)

# bottom: dtype / model note
box(2, 12, 96, 11.5, "MODEL & PRECISION (fixed)", [
    "encoder = V-JEPA 2 ViT-g (1.01B, frozen)   predictor = V-JEPA 2-AC (305M)   -- image 256x256, patch->256 tokens/frame",
    "dtype = bf16   device = RTX 3090 (24 GB)   objective = mean-L1 in layer-norm'd latent space   -- gripper axis frozen during V-JEPA reach",
    "we plan the COARSE arm motion with V-JEPA+CEM; SCRIPTED primitives do the gripper (close/lift/open); success judged only from privileged sim state",
], "#eeeeee", fs=8.6)

# legend of the loop nesting
ax.text(50, 3.2, "loop nesting:  BENCHMARK > TRIAL > TASK(stages) > MPC time-step (x 4/10/4 ...) > CEM (x10 iters, x200 samples, x T=2 rollout)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#333")

fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
