# Experimental head/wrist → SMPL-X → robot path

Independent Nymeria pilot, 2026-10-13. The existing three viewer pipelines are unchanged.

## What ran

Nymeria Xsens anatomical head and bilateral wrist poses, 450 frames at 30 Hz, were passed to EgoAllo with wrist guidance. Pelvis, ankles, other body joints and foot contacts were withheld. The recording is `20230803_s0_robert_howard_act2_kobpfk`, starting at 30-Hz frame 17684 (carry jacket to wardrobe). The source video is the matching 15-second excerpt.

1. `prepare_nymeria.py`: repair the existing Xsens clock, select a uniform window, split sparse observations from held-out body data. Requires `--source`, `--start-frame`, `--frames`, `--output`.
2. `run_egoallo.py`: official EgoAllo checkpoint, seed 20261013, `aria_wrist_only`, inner and post guidance. A neutral SMPL-H anatomical head-to-eye offset approximates CPF. This is **not** calibrated Aria camera input.
3. `transfer_smplx.py`: fit one constant SMPL-X shape to the average predicted SMPL-H rest skeleton, then transfer shared local body and finger rotations. Generate actual 10475-vertex SMPL-X meshes and a 55-joint canonical clip. This is an approximate pose transfer, not the official surface-correspondence transfer.
4. `prepare_robot_jobs.py`: make independent H2/A3 jobs for the existing `fullbody` backend. The shared AlphaMotion + contact + body-coordination implementation is reused without robot-specific solver parameters. This pilot consequently uses the fullbody backend's 100+100 refinement, **not** the legacy Nymeria four-endpoint solver.
5. `export_wuji_robot.py`: map SMPL-X 21-point hand landmarks through Wuji SDK, preserve native body playback, attach each 20-DoF hand by a fixed wrist transform. H2 uses its existing 58-mm mount; A3 uses the native hand mesh's approximately 48-mm proximal flange. Mounts are visual integration metadata, not mechanically certified hardware designs.
6. `build_wuji_urdf.py`: export portable H2/A3 URDFs and localized meshes, each with 40 finger joints.
7. `render_human.py`, `render_robot.py`: 30-Hz kinematic videos; no simulation, motion smoothing, or visual ground lift.
8. `validate_results.py`: report transfer error, source-wrist error, held-out comparisons, final robot positions, and acceleration diagnostics. No acceptance gate is enabled.

The saved Nymeria Xsens data has **no finger observations**. Finger articulation in these videos comes from EgoAllo's inferred prior. Wuji mapping is real, but this is not evidence of recovering the actor's true grasp.

## Coordinate contract

EgoAllo world is Z-up; SMPL local joint frames and rest offsets are Y-up. World positions use `C @ p`, global joint orientations use **`C @ R`**, and local rest offsets remain unchanged. `C=[[0,1,0],[0,0,1],[1,0,0]]`. Do not conjugate rotations and also rotate the rest skeleton: that produces a sideways neutral descriptor. The final fix preserves every human world position exactly and passes canonical FK validation.

## Environment and reproducibility

Private work directory on the A6000 host: `/home/ubuntu/arenalab-experiments/egoallo`.
Set `ARENA_EGO_WORKDIR` to that artifact directory when running these saved scripts elsewhere; defaults to the script directory. Add the repository `src` to `PYTHONPATH`.

EgoAllo uses an isolated Python 3.12 environment, torch 2.7.1, JAX 0.6.1 on CPU for guidance, official upstream revision `0c20ae8c8d48be4ff682803ea523bcb00a6a708d`, and jaxls revision `e19f32539a1416f65257c3e9b0bbd843f469d924`. Rendering uses PyOpenGL 3.1.10 and pyrender 0.1.45 with the NumPy `infty` compatibility alias. The production Python 3.10 environment runs AlphaMotion and Wuji. Exact environment is retained in `environment.txt` in the private work directory.

Required private files: SMPL-H `model.npz`, SMPL-X `SMPLX_NEUTRAL.npz`, the official EgoAllo April13 checkpoint directory, and Nymeria observations. Licensed models/checkpoints/data are not committed. `prepare_robot_jobs.py` reads the existing cloud catalog; job JSONs can instead be provided manually using the normal repository job contract.

## Original baseline outcome (superseded by observed-wrist refinement)

The end-to-end path ran on H2 and A3 with matching 450-frame timelines. It is a research result, **not an accuracy-qualified manipulation pipeline**.

- SMPL-X wrist error to source: mean 19.9 mm, maximum 40.1 mm.
- SMPL-H predictions vary body shape over time. Fixed-shape SMPL-X conversion introduces additional geometric differences (all-joint mean 21.8 mm).
- SMPL-X mesh minimum world height is -56.9 mm. Its inferred lower body is not guaranteed ground-consistent.
- Final H2/A3 wrist position differences to the unscaled SMPL-X world trajectory average 11.4/16.3 cm. Robot morphology and the existing posture/contact objectives affect these numbers; they are not a calibrated TCP task-tracking metric. The reused fullbody refiner does not impose strict world-space wrist tracking.
- Nymeria pelvis/ankle comparisons have different anatomical landmark definitions and no subject-specific calibration; treat them as diagnostics, not ground-truth accuracy claims.
- Final root vertical acceleration still has peaks, approximately 11.8 m/s² for H2 and 26.8 m/s² for A3. No dynamics acceptance is claimed.

Before production manipulation use: calibrate camera/head and anatomical frames, enforce a consistent human shape, preserve original wrist observations through a shared task-space refinement alongside reconstructed pelvis/feet, and add actual finger observations for grasp retargeting. Failed trials and immutable AlphaMotion priors are retained in the private work directory.

## Observed-wrist refinement, 2026-10-13

`refine_observed.py` now solves native joint angles, root position and root rotation together. Original Nymeria bilateral wrist position and palm orientation remain the task targets. SMPL-X supplies knee flexion, pelvis vertical motion and foot motion priors. Foot support is a soft confidence inferred from height and speed; no all-corners-flat or zero-slip lock and no final root-height snap is applied.

The earlier locomotion fix in `run_contact_knee_refiner.py` restored source knee flexion with coupled hip/ankle compensation. This experiment reuses that principle within the joint solve: a robot's physical hinge limits and FK calibrate a signed knee target from human flexion. It does not copy robot-name-specific axis assumptions. An unsigned angle objective admitted hyperextension and caused jumps in v1; that candidate is preserved and rejected. V2 fixes the branch. V3 additionally allows a 2-mm position and 0.035-radian orientation objective deadband, leaving room for temporal smoothing; these are objective parameters, not acceptance thresholds. User acceptance remains 1 cm / 15 degrees on every frame.

Both H2 and A3 run the identical policy on two 450-frame, 30-Hz excerpts: carry jacket (start 17684) and crouch/unload washer (start 10664), from the same Nymeria recording. True pelvis/feet from Xsens remain withheld. `prepare_nymeria.py --clip` stores each clip's provenance.

`validate_observed.py` reloads the saved final motor angles into the original MuJoCo model and checks wrist poses, all native foot-mesh vertices, joint limits and derivatives. `diagnose_knees.py` separates SMPL-X, raw AlphaMotion, physical projection and final refinement. `assemble_review.py` creates synchronized videos and decodes all 450 frames. Detailed sample results are retained alongside videos in `outputs/nymeria-observed-refinement/validation-summary.json` in the task workspace, not embedded into production policy.

Run refinement with the production Python environment, repository `src` on `PYTHONPATH`, `OPENBLAS_NUM_THREADS=1`, and `ARENA_EGO_WORKDIR` pointing to the private clip directory. Run rendering with the isolated EgoAllo environment. The refinement budget is 1600 L-BFGS-B iterations; reaching that budget is reported honestly as an optimizer iteration limit, not convergence. No default production acceptance gate is introduced.

Limits: these are kinematic sample checks, not simulation or motor-torque validation. Deep human flexion may exceed the native robot's range and is clipped through physical calibration. Self-collision, friction and balance are not certified. The reconstructed human mesh itself can penetrate the floor; robot floor checks apply to final native robot meshes. Finger recovery is deferred; the existing inferred hand prior is retained and explicitly labeled in videos. The existing production viewer and three production pipelines are unchanged.

Final v3 native-FK results (maxima across both wrists and all 450 frames):

| Clip | Robot | Wrist position, cm | Wrist orientation, deg | Root vertical acceleration before → after, m/s² | Joint acceleration before → after, rad/s² |
|---|---|---:|---:|---:|---:|
| Carry jacket | H2 | 0.239 | 4.251 | 11.816 → 3.609 | 145.255 → 73.690 |
| Carry jacket | A3 | 0.211 | 3.307 | 26.809 → 3.348 | 291.557 → 46.526 |
| Crouch/unload washer | H2 | 0.228 | 5.737 | 52.301 → 4.021 | 135.050 → 107.068 |
| Crouch/unload washer | A3 | 0.211 | 5.531 | 64.677 → 3.918 | 294.095 → 104.305 |

All four clips satisfy the user's 1 cm / 15 degree wrist tolerance on every frame. Worst native foot penetration is 0.022 mm (H2 carry jacket); the other three minima are positive. No native joint-limit violation exceeds floating-point noise. Crouch peak knee flexion is approximately 141 degrees on H2 and 143 on A3, versus approximately 160 degrees for SMPL-X. Ten delivered videos (source, human, both robots, combined comparison for each clip) decode to exactly 450 frames at 30 Hz. All four solves reached the 1600-iteration budget rather than optimizer convergence; the finite sample checks above do not establish zero-shot generalization or dynamic feasibility.
