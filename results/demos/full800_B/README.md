# Demo reel -- full800_B benchmark rollouts

Real rollouts from the paper-faithful benchmark run (**samples 800, horizon 1, maxnorm 0.075, camera
B_closer**), reproduced deterministically from the logs (`scripts/make_demo_gifs.py`) -- no re-run of
V-JEPA, so they match exactly what happened. One **HIT** (success) and one **MISS** (failure) per
(task, object), labeled with the outcome, final error, and per-frame phase/held/tilt.

## Grasp (V-JEPA plans the reach; scripted close + lift)

| | HIT | MISS |
|---|---|---|
| **cup** (group: 40% @6cm) | ![grasp cup hit](grasp_cup_HIT.gif) | ![grasp cup miss](grasp_cup_MISS.gif) |
| **box** (group: 12% @6cm) | ![grasp box hit](grasp_box_HIT.gif) | ![grasp box miss](grasp_box_MISS.gif) |

Grasp misses are typically the object *tipping* or *slipping during settle* even when the reach lands
within 1-3 cm -- the bottleneck is the sim close-and-lift mechanics, not the planner.

## Reach-with-object (object starts held; V-JEPA carries it to a goal)

| | HIT | MISS |
|---|---|---|
| **cup** (group: 98% @10cm, beats paper 75%) | ![rwo cup hit](reach_with_object_cup_HIT.gif) | ![rwo cup miss](reach_with_object_cup_MISS.gif) |
| **box** (group: 94% @10cm, beats paper 75%) | ![rwo box hit](reach_with_object_box_HIT.gif) | ![rwo box miss](reach_with_object_box_MISS.gif) |

Reach-with-object is our strongest task -- the pure "move a held object to a goal image" skill
exceeds the paper's real-robot success rate.

_Regenerate: `python scripts/make_demo_gifs.py` (auto-picks 1 HIT + 1 MISS per completed group)._
