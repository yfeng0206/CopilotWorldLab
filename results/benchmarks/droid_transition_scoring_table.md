| metric | value | meaning |
| --- | --- | --- |
| transitions (n) | 300 | real DROID (image_t, state_t) -> (image_t+H) pairs |
| rank_frac | 0.820 | mean fraction of 32 random negatives the true xyz action beats (chance 0.5) |
| null rank_frac | 0.486 | same, but goal from a different episode (image-conditioning control) |
| conditioning gap | +0.334 | rank_frac - null; the goal-image effect |
| top1_acc | 0.320 | fraction of transitions where the true action beats ALL negatives |
| gap_z | +1.45 | mean (neg energy - true energy) / std, effect size |
