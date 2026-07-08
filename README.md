# CopilotWorldLab

A latent video world model (V-JEPA 2-AC) plans coarse robot-arm motion by minimizing distance to a
goal image, and the model's own predictive energy is proposed as the confidence signal that hands off
to a classical, vision-only precise controller. This repository is the Stage-1 simulation substrate.
The design rationale is in [`docs/DESIGN.md`](docs/DESIGN.md).

<p align="center">
  <img src="results/demos/full800_B/reach_with_object_cup_HIT.gif" width="360" alt="reach-with-object cup, success">
  <img src="results/demos/full800_B/grasp_cup_HIT.gif" width="360" alt="grasp cup, success">
</p>
<p align="center"><em>V-JEPA 2-AC driving a Franka in MuJoCo, planning to goal images.
More clips: <a href="results/demos/full800_B/">demo reel</a>.</em></p>

## What this is

We reproduce the planning result from *V-JEPA 2* (Assran et al., arXiv:2506.09985, §4) in our own
MuJoCo environment, then evaluate it as a coarse controller across a fixed benchmark. The model is
frozen; motion comes entirely from model-predictive control (greedy horizon-1 CEM) toward image
goals. Gripper open/close is scripted at fixed stage boundaries, so the reported numbers isolate
V-JEPA's spatial planning. Success is judged from hidden simulator state, not the latent energy.

## Benchmark

Five tasks x two objects (a rim-graspable cup and a rigid box) x 50 fixed scenarios = 500 rollouts,
scored at paper-faithful settings (800 CEM samples, 10 refinement steps, horizon 1, maxnorm 0.075).
Every configuration runs on the same saved bundles under `tasks/`, so results are reproducible and
directly comparable.

| task | object starts | V-JEPA plans | our success | paper (Table 3) |
|---|---|---|---|---|
| grasp | on table | reach to grasp pose | cup 38% / box 10% @6cm | 65% / 25% |
| reach_with_object | held | carry to a goal | cup 98% / box 96% @10cm | 75% / 75% |
| grasp_and_reach | on table | grasp, then carry (2 goals) | cup 18% / box 4% @10cm | custom |
| pick_place | on table | grasp, vicinity, place (4/10/4) | running | 80% / 65% |
| place_with_object | held | carry to zone, place (2 goals) | running | custom |

reach_with_object exceeds the paper's real-robot rate. The table is a hard contact, so a light object
cannot be pushed into it; if the arm drives the gripper into the tabletop the trial fails outright
rather than tunneling through. Grasp misses are mostly a few-cm reach error before the object tips or
slips. Live results and precision curves:
[`results/benchmarks/full800_B_progress/`](results/benchmarks/full800_B_progress).

## How motion is produced

At each control step the model renders the current frame, runs CEM to find the single next
end-effector action whose predicted next latent is closest to the goal image, executes it, and
replans. The energy landscape is locally convex near the goal (paper Fig. 9), so greedy descent walks
the arm to the target — a learned form of visual servoing. Compositional tasks follow a fixed
sub-goal schedule switched by time index (pick-and-place: 4/10/4), reproduced from the paper. Full
explanation with clips: [`results/demos/full800_B/README.md`](results/demos/full800_B/README.md).

## Roadmap

| Phase | Content | State |
|---|---|---|
| 0 | Load V-JEPA 2-AC, build a paper-faithful MuJoCo env, reproduce the world model | done |
| 1 | Fixed-bundle closed-loop benchmark (5 tasks x cup/box x 50) | current |
| 2 | POV/wrist CNN coarse-to-fine handoff | planned |
| 3 | Third- + first-person cross-attention latent | planned |
| 4 | Unified cross-view latent | planned |

Earlier reproductions (energy landscape, DROID transition scoring, closed-loop CEM pilot, camera
ablation) are written up under [`docs/experiments/`](docs/experiments).

## Repository layout

```
src/envs/franka_build.py          Franka Panda + Robotiq 2F-85 scene (cup/box, place zone, distractors)
src/envs/franka_droid_env.py      FrankaDroidEnv: 7-DoF EE control via differential IK
src/bench/                        Task-bundle schema, hidden success functions, thresholds/gates
src/world_model/vjepa2_wrapper.py V-JEPA 2-AC control-loop scaffold
scripts/generate_task_bundles.py  Scripted expert -> fixed task bundles under tasks/
scripts/run_closed_loop_benchmark.py  Closed-loop CEM-MPC benchmark; loads bundles, hidden-state success
scripts/make_demo_gifs.py         Labeled HIT/MISS rollout GIFs, reproduced from logs
scripts/replay_from_log.py        Reproduce any trial in 3D from the log (deterministic, no GPU)
tasks/                            Fixed task bundles (gitignored; regenerate with the generator)
tests/                            Geometry, env, grasp physics, success, thresholds
docs/                             DESIGN, architecture, experiments/, research_log, lessons_learned
```

## Setup

Windows 11, RTX 3090 (24 GB), Python 3.11. PyTorch from the CUDA 12.4 wheel index; the rest from
`requirements.txt`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\Activate.ps1
python -m pytest -q
python scripts\download_checkpoints.py        # V-JEPA 2-AC checkpoint (~11.8 GB)
```

The V-JEPA scripts import `third_party/vjepa2`; the env needs MuJoCo Menagerie (both gitignored):

```powershell
git clone https://github.com/facebookresearch/vjepa2 third_party/vjepa2
git -C third_party/vjepa2 checkout 204698b
git clone --depth 1 --filter=blob:none --sparse `
  https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda robotiq_2f85
```

## Links

| | |
|---|---|
| Design, novelty, roadmap | [`docs/DESIGN.md`](docs/DESIGN.md) |
| Architecture + flowcharts | [`docs/architecture.md`](docs/architecture.md) |
| Benchmark methodology | [`docs/experiments/closed_loop_benchmark.md`](docs/experiments/closed_loop_benchmark.md) |
| Experiments index | [`docs/experiments/`](docs/experiments) |
| Research log + bibliography | [`docs/research_log.md`](docs/research_log.md) |
| Lessons learned | [`docs/lessons_learned.md`](docs/lessons_learned.md) |

## References

- Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
  Planning.* arXiv:2506.09985 (2025).
- Khazatsky et al. *DROID: A Large-Scale In-the-Wild Robot Manipulation Dataset.* arXiv:2403.12945 (2024).
- Todorov et al. *MuJoCo: A physics engine for model-based control.* IROS 2012.

A context-annotated bibliography is in [`docs/research_log.md`](docs/research_log.md#paper-bibliography).
