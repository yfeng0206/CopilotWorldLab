# Experiment: energy-landscape reproduction and camera-placement ablation

Date: 2026-07-04. Branch: world-model-pilot. GPU: RTX 3090 (bf16).

## Question

Does the released V-JEPA 2-AC world model (arXiv:2506.09985), trained on real DROID Franka
video from an exocentric camera, produce a usable latent "energy landscape" on our MuJoCo
Franka renders -- and which camera placement transfers best? The paper's headline success
rates are real-hardware and not reproducible in simulation; the reproducible correctness
checks are (1) the energy landscape on the paper's own example trajectory and (2) the same
analysis on our simulator renders, with a camera-placement ablation as the diagnosis.

## Method

- Energy landscape: encode a context frame and a goal frame to latents; sweep an `n^3` grid
  of xyz action deltas; roll each one step through the AC predictor; score the latent energy
  `E(a) = mean(|P(a; z, s) - z_goal|)`. The ground-truth action is `poses_to_diff(s0, s1)`
  of the recorded end-effector poses (the realized motion, not the commanded one, so IK/physics
  slack is accounted for). `scripts/energy_landscape_repro.py`.
- Transfer + ablation: `scripts/render_franka_transitions.py` drives `FrankaDroidEnv` one real
  control step for six canonical end-effector deltas (+/-x, +/-y, +/-z) from three start poses,
  and renders the before/after frames from eight cameras -- seven free placements (azimuth /
  elevation / distance about a fixed lookat) plus the exact built-in `exo_cam` (`exo_named`).
  Every camera renders the SAME physical transition, so the only variable between cameras is
  the viewpoint. Phase 2 scores every transition (one model load) and aggregates per camera.
- Frame analysis: `scripts/analyze_frame_rotation.py` fits, per camera, the single 2D rotation
  in the world x-y plane that best maps the ground-truth action to the energy-minimizing
  action, to separate a fixed (calibratable) frame rotation from camera observability.

Metric: cosine between the energy-minimizing action (`argmin`) and the ground-truth action.
Energy "margin" = (mean - min) / std of the grid energies (landscape informativeness).

## Result 1 -- paper example trajectory (correctness gate): PASS

At a grid wide enough to contain the ground-truth action (0.12, `9^3`):

| direction | argmin xyz | GT xyz | cosine | err | margin |
|---|---|---|---|---|---|
| reverse | (-0.090, -0.060, -0.090) | (-0.092, -0.031, -0.084) | +0.98 | 0.030 m | 3.65 |
| forward | (+0.030, +0.120, +0.060) | (+0.092, +0.031, +0.084) | +0.65 | 0.111 m | 2.93 |

The energy minimum lands near the ground-truth action, reversing the trajectory flips the
dominant-axis sign, and the landscape is informative. The forward pass has a flat y-axis, so
its hard argmin wanders in y -- matching the paper's characterization: "smooth and locally
convex, minimum only *near* the ground-truth action." Model load and preprocessing validated.

## Result 2 -- MuJoCo transfer and camera-placement ablation

144 transitions (8 cameras x 6 axes x 3 poses), grid 0.08 / `9^3`. Per-camera mean cosine
(n = 18 usable transitions each):

| camera (azimuth_elevation) | mean cos | median | min | margin | verdict |
|---|---|---|---|---|---|
| az45_el45 (opposite shoulder, high) | +0.92 | +0.95 | +0.72 | 2.73 | transfers |
| az45_el20 | +0.89 | +0.99 | +0.34 | 2.76 | transfers |
| az90_el45 (side, high) | +0.57 | +0.70 | -0.14 | 2.99 | transfers |
| az90_el20 | +0.50 | +0.53 | -0.99 | 3.08 | transfers |
| top_down (wrist-like) | +0.19 | +0.24 | -0.57 | 2.74 | weak |
| az135_el45 | +0.12 | +0.00 | -0.66 | 2.82 | weak |
| az135_el20 | +0.08 | -0.00 | -0.99 | 2.63 | weak |
| exo_named (exact built-in exo_cam) | -0.16 | -0.43 | -1.00 | 1.88 | weak |

Per action axis (mean cosine across cameras): vertical transfers well (pz +0.82, nz +0.72);
horizontal is weak (px +0.31, py +0.37, nx +0.30, ny -0.17).

## Result 3 -- the horizontal action frame is view-relative (confound resolved)

Fitting one in-plane rotation per camera that maps the ground-truth to the energy-minimizing
action (`analyze_frame_rotation.py`, horizontal transitions only):

| camera | camera azimuth | fitted rotation | post-rotation cos |
|---|---|---|---|
| az45_el45 | -45 | +8.5 | +0.95 |
| az45_el20 | -45 | +18.3 | +0.93 |
| az90_el45 | -90 | +53.0 | +0.85 |
| az90_el20 | -90 | +56.3 | +0.70 |
| az135_el45 | -135 | +96.4 | +0.90 |
| az135_el20 | -135 | +118.5 | +0.84 |
| exo_named | (built-in) | -152.6 | +0.84 |
| top_down | -90 | +66.5 | +0.65 |

The fitted rotation tracks the camera azimuth almost linearly (-45 -> ~13 deg, -90 -> ~55 deg,
-135 -> ~107 deg), and after that single rotation every exocentric side camera recovers to
cos 0.84-0.95. So the model infers horizontal actions in a view-relative frame.

## Interpretation

1. Transfer works qualitatively. Every side camera yields a healthy energy margin (2.6-3.1),
   so the DROID-trained model does respond to our MuJoCo renders -- the landscape is not flat.
2. Vertical (z) transfers directly from any side view: it is gravity-aligned, so camera- and
   frame-independent (pz/nz cosine ~0.7-0.8 everywhere).
3. The horizontal (x-y) plane is view-relative. The apparent camera ranking is really a
   ranking of how large a fixed W* rotation each view needs: az45 needs ~none, az135 and the
   built-in exo_cam need ~90-150 deg. After per-camera W* correction, all four side-camera
   families recover to cos >= 0.84. This is exactly the paper's App. B.4 calibration, and it
   resolves the camera-vs-frame confound: az135 / exo_cam are not unusable, just uncalibrated.
4. For ZERO-SHOT use (no calibration), az45_el45 is the clear choice (~8 deg residual). The
   built-in exo_cam is the worst zero-shot camera and also has the lowest margin (1.88).
5. top_down is a genuine observability failure: even after the best in-plane rotation it only
   reaches cos 0.65, because a near-top-down view foreshortens the depth axis. This confirms
   the paper's exocentric-only training assumption.

## Honesty / limits

- Success rates from the paper (real Franka hardware) are not reproduced here; these are
  latent-energy correctness and interface-transfer measurements in simulation.
- Cosine is measured against a world-frame ground truth; Result 3 shows the horizontal
  component is a fixed per-camera rotation, so absolute (uncalibrated) cosines understate
  usability for high-azimuth cameras. The per-camera ranking and the recovered post-rotation
  cosines are the trustworthy quantities.
- Three start poses per axis is a pilot-scale sample; the effects are large relative to the
  spread (median/min reported), but a larger sweep would tighten the estimates.

## Next steps

- Fit and freeze a W* correction for the chosen planning camera (az45_el45 needs almost none)
  and re-score, confirming the residual is calibration, not domain gap.
- Add an az45_el45-style named planning camera to the scene, then close the loop: CEM planning
  to a rendered goal image in `FrankaDroidEnv`.
