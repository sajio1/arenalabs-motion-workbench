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

## Direct torso-prior experiment (v4/v5)

`body_frames.py` identifies the chest as the common ancestor of the two wrist chains and the pelvis as the common ancestor of the two ankle chains. Bilateral shoulder/hip anchors and neutral FK define fixed anatomical frames. The same topology-based code is used for both robots; no robot-name branch selects solver behavior. Human targets use SMPL-X spine3 (joint 9) and pelvis (joint 0), transformed on the world side from Z-up to canonical Y-up.

Two soft objectives are added: chest world orientation and chest orientation relative to pelvis. Each uses an SO(3) angular residual, a 0.0873-radian deadband and a 0.25-radian scale. No absolute human head-height target, elbow target, or finger change is added. The human body targets remain reconstructed priors, not additional observations.

V4 added only these body terms to the v3 policy. It reduced body-orientation error on all four cases but increased some acceleration peaks. V5 retains those terms and strengthens joint second-difference smoothing (scale 0.025 → 0.018 radians per frame squared) and root-correction second-difference smoothing (0.0015 → 0.0012 metres per frame squared). Both policies and all candidate outputs are retained in the private work directory. V3 and V4 directories are immutable comparison baselines; v5 is the current experimental candidate.

Correction to the initial mechanical audit: the loaded H2 model DOES have waist pitch, approximately -25 to +30 degrees, in addition to yaw and roll. The descriptor merges `torso_link` into a node named `waist_roll_link`; the original inspection incorrectly used that node name as the native frame and omitted pitch. Resolve through `build_merge`'s representative-frame mapping before inspecting a native chest chain. The loaded A3 model has waist pitch approximately -28 to +24 degrees. The refiner itself already included the merged pitch DoF; the earlier explanation that H2 lacked pitch was wrong. Residual forward tilt needs an objective/reachability diagnosis, not that explanation.

`validate_torso.py` reports anatomical-frame rotation errors, forward tilt and derivative metrics against v3, and checks saved chest rotations against physical-axis FK. `validate_observed.py` independently repeats native wrist/foot/limit checks on the final motion. `assemble_torso_review.py` creates v3/v5 comparison videos and fully decodes all frames. Samples and detailed reports are saved under the task's `outputs/nymeria-torso-prior`; production pipelines remain unchanged.

Final v5 vs v3 sample results:

| Clip / robot | Mean absolute forward-tilt error to SMPL-X, deg | Max wrist error, cm / deg | Peak joint acceleration, rad/s² |
|---|---:|---:|---:|
| Carry jacket / H2 | 14.66 → 9.54 | 0.233 / 4.50 | 73.69 → 70.66 |
| Carry jacket / A3 | 5.63 → 2.51 | 0.217 / 3.85 | 46.53 → 47.52 |
| Crouch / H2 | 19.40 → 11.56 | 0.246 / 6.80 | 107.07 → 92.33 |
| Crouch / A3 | 5.89 → 4.01 | 0.221 / 6.93 | 104.31 → 95.14 |

All 1800 robot frames pass 1 cm / 15 degree bilateral wrist tolerance. Worst native foot penetration is 0.332 mm on crouch/H2; all other minimum heights are positive. Native joint-limit excess is under 8e-8 radians. The A3 walking peak joint acceleration rises by approximately 2%; H2 walking peak root acceleration also rises from 3.61 to 3.93 m/s². Thus the candidate improves average body tracking but does not improve every temporal metric. H2 still shows appreciable extra forward tilt and constrained lumbar pose: this is a partial improvement, not exact human-pose recovery. V5 also exhausts the 1600-iteration budget; no convergence or physical-feasibility claim is made.

## Posture/morphology experiment (v6/v7)

Controlled v5 ablations on carry-jacket/H2 found that removing wrist orientation barely improved torso tracking and increased maximum wrist orientation error to 49 degrees. Removing root-height or model priors had little effect. Strengthening the world-chest objective (scale 0.25 to 0.08 radians) improved body tracking, but v6 increased extra foot movement. V6 is retained as a rejected intermediate, not the recommended result.

V7 retains the stronger chest term and adds two shared objectives. One optimized, clip-wide knee-flexion offset is shared by both legs, softly regularized at 0.3 radians and bounded to ±0.5 radians. Signed native hinge calibration maps the offset to each robot's valid bending branch and clips to its mechanical range. This preserves the reference knee time pattern except where range clipping is necessary, while allowing stature adaptation. The offset is a solved variable, never a robot-name parameter. A horizontal foot-correction-velocity objective penalizes added motion relative to the human prior, with a 3.3-mm/frame deadband and scale at the fixed 30-Hz timebase. It does not demand zero foot velocity or perfectly fixed heel/toe contact.

| Clip / robot | Mean absolute forward-tilt error v5 → v7, deg | Solved knee offset, deg | Max wrist error, cm / deg |
|---|---:|---:|---:|
| Carry jacket / H2 | 9.54 → 3.81 | 11.78 | 0.297 / 5.31 |
| Carry jacket / A3 | 2.51 → 2.22 | 0.55 | 0.220 / 3.83 |
| Crouch / H2 | 11.56 → 5.56 | 8.12 | 0.272 / 7.06 |
| Crouch / A3 | 4.01 → 2.86 | 0.22 | 0.220 / 7.10 |

The ankle-speed proxy during low/slow reconstructed-human phases improves in all four cases (p95: 60.09→12.67, 20.09→12.67, 26.15→15.30, 21.70→13.95 cm/s). This is an ankle diagnostic, not measured material-contact slip. Crouch/H2 mean horizontal ankle deviation from the human prior worsens 2.90→4.29 cm: body adaptation trades exact human foot placement for torso posture. Peak joint acceleration changes 70.66→73.98, 47.52→48.51, 92.33→96.76, 95.14→94.08 rad/s² respectively. Do not claim universal temporal improvement or physical feasibility. All runs use the same 1600-iteration budget and reach it; no convergence claim is made. Raw AlphaMotion hashes and all prior versions are retained.

`validate_native_chest.py` checks descriptor chest matrices against the actual merged representative native body, avoiding the earlier node-name inspection error. `validate_foot_prior.py` reports the foot-placement/motion tradeoff. `validate_torso.py` now defaults to v5 as the before baseline; `ARENA_BASELINE_FOLDER` can select an older comparison. Current review artifacts are in the task's `outputs/nymeria-posture-priority` folder. Production code is unchanged.

Final v7 native validation: all 1800 robot frames pass the user's bilateral wrist tolerance. Worst native mesh penetration is 0.4033 mm (carry-jacket/H2); crouch/H2 is 0.0932 mm, and A3 minima remain above ground. Native joint-limit excess is below 8e-8 radians. Chest rotation fields agree with native representative bodies within 2.1e-6 degrees in every case. Four final robot videos and two before/after composites were fully decoded: each is 450 frames at 30 Hz. This validates these saved kinematic samples, not torque, balance, self-collision, or unseen-motion performance.
