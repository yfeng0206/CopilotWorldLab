# Sample fixed task bundles

This directory contains a small, tracked sample of the fixed-bundle benchmark data: one scenario for each task/object pair.

Full benchmark bundles are generated under the gitignored `tasks/` directory. These samples are for peer review, data inspection, and smoke testing only; they are not a benchmark result.

## Contents

| Task | Objects | Sample IDs |
|---|---|---|
| `grasp` | `cup`, `box` | `grasp_cup_00`, `grasp_box_00` |
| `reach_with_object` | `cup`, `box` | `reach_with_object_cup_00`, `reach_with_object_box_00` |
| `grasp_and_reach` | `cup`, `box` | `grasp_and_reach_cup_00`, `grasp_and_reach_box_00` |
| `pick_place` | `cup`, `box` | `pick_place_cup_00`, `pick_place_box_00` |

Each bundle includes `meta.json`, `start.png`, `goal.png`, optional sub-goal images (`goal_1.png`, `goal_2.png`), `arrays.npz`, and `contact_sheet.png`.

## Inspect

```powershell
python scripts\inspect_task_viewer.py --object cup
python scripts\inspect_task_viewer.py --object box
```

Use `N` / `B` or right/left arrows to step through stages in the viewer.

## Smoke run

```powershell
python scripts\run_closed_loop_benchmark.py --bundles examples\task_bundles --tasks grasp reach_with_object grasp_and_reach pick_place --objects cup box --trials 1 --samples 100 --tag sample_smoke
```

Generate the full local benchmark set with:

```powershell
python scripts\generate_task_bundles.py --tasks grasp reach_with_object grasp_and_reach pick_place --objects cup box --trials 50 --tasks-dir tasks
```
