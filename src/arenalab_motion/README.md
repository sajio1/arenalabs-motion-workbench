# Private full-body GPU motion backend

This package ports the pre-fixed-hand-tolerance HiPhi scaled v3 algorithm. It
uses the same 600 hand coefficient, 60 soft and 20 polish iteration caps,
nonlinear floor restoration, FP64 residual/Jacobian assembly, temporal block
solve, StableIPM and CUDA Graph execution. Iteration caps are not convergence
claims. All robots/actions share this numerical policy.

The surface term preserves sampled **actual robot visual-mesh positions and
normals around the immutable decoded prior**. It is not a learned human/robot
surface correspondence. This is kinematic retargeting, without WBC, dynamics,
object-contact or self-collision certification. Current native rigs require one
floating root, scalar hinges and no mechanical equality constraints. Visual
mesh, box and cylinder geometry are supported; other primitives fail explicitly.

## Invocation

Install this repository with its runtime dependencies. AlphaMotion code is part
of this repository; no research checkout is imported at runtime. Tested locally
with Python 3.11, PyTorch 2.7.0 CUDA 12.8, NumPy 1.26.4 and MuJoCo 3.4.0.
The deployment must smoke-test its own driver/MuJoCo environment.

```sh
python -m arenalab_motion.runner --request request.json
```

The JSON accepts `arenalabs.private-motion-request.v1` or `arenalabs.motion.v1`:

```json
{
  "schema": "arenalabs.motion.v1",
  "robot_id": "customer_robot",
  "row": {
    "name": "customer_robot",
    "xml": "/private/robot/model.xml",
    "body": "floating_root_body",
    "endpoints": {
      "left_wrist": "left_wrist_body", "right_wrist": "right_wrist_body",
      "left_ankle": "left_ankle_body", "right_ankle": "right_ankle_body"
    },
    "palm_frames": {
      "left_wrist": {"distal": [1, 0, 0], "normal": [0, 0, 1]},
      "right_wrist": {"distal": [1, 0, 0], "normal": [0, 0, 1]}
    }
  },
  "source_work": "/private/source/normalized",
  "output_dir": "/private/results/new-immutable-version",
  "runtime_config": "/private/config/runtime.json",
  "scale_mode": "preserve"
}
```

Example palm axes are placeholders, not default calibration. Supply measured or
explicitly declared axes for the actual robot. `semantic_metadata.semantics`
can map root/chest and bilateral shoulder/elbow/wrist/hip/knee/ankle to native
body names. Without it, `ensure_robot_metadata` uses confirmed endpoint ancestry
and physical anatomical joint labels, or unambiguous neutral geometry. It fails
on ambiguity and never selects behavior by a robot/action identifier.

Runtime JSON has required `weights` and `cache` directories, optional `hf_home`
and `device` (default `cuda:0`). Relative paths resolve beside the runtime JSON.
`weights` must already contain the requested Prime checkpoint's `config.json`
and `model.safetensors`. `ARENA_MOTION_RUNTIME` can select the runtime JSON.
Checkpoint SHA values are saved in the immutable prior receipt. Downloads are
explicitly disabled. Scheduler workers should set `CUDA_VISIBLE_DEVICES` and use
`cuda:0` inside that worker; the package does not schedule other workers itself.

## Source boundary and coordinates

The source adapter must first produce `native_source.npz` and
`input_manifest.json`. Raw supplied SMPL/H/X are dispatched by their real format
upstream. With missing source body, the upstream three-point EgoAllo path
reconstructs native SMPL-H. This backend never replaces a supplied human body
with six-point completion. Every supplied body joint is visible to the encoder.

Required arrays are `names`, `parents`, `rest_offsets_cm`,
`full_body_position_yup_m[T,J,3]`, `full_body_rotation_yup[T,J,3,3]`,
`timestamps_s[T]`, `exact_fps`, `floor_z_m`,
`wrist_position_world_m[T,2,3]`, `sole_low_z_m[T,2]`, and separately derived
physical `palm_forward[T,2,3]`/`palm_normal[T,2,3]`. The complete body is a
single-root topological tree with canonical semantic names. World-native points
are Z-up metres; encoding uses Y-up metres and centimetre rest offsets.
`C=[[0,1,0],[0,0,1],[1,0,0]]`; row points convert by `p_z=p_y@C`.
Global transfer rotations express both local and world axes in canonical Y-up;
physical palms are independent and must not be guessed from those matrices.

Optional precomputed `source_body_position_world_m`,
`source_body_rotation_world_transfer`, `bone_directions_world`,
`six_position_zup_m` and `reference_world_R_zup` retain exact saved baseline
arrays. Only contiguous nominal 30 Hz clocks are supported; original timestamps
and source-frame indices are retained (including HiPhi 30.000300003 Hz sampling).
The source body manifest must describe whether its body is fitted/reconstructed.

`preserve` leaves current input geometry untouched. Explicit `match_robot` fits
one scalar to eight limb lengths, anchors at first pelvis projected to the
source ground, and scales human/objects/operation targets together. It creates
new normalized source files, preserves originals, and rejects double scaling.
This is a normalized-size task, not fidelity to the original fixed-size object.
A generic mesh/pose consumer must use the saved similarity; a scaled human is
not represented by changed translation alone with an unscaled SMPL model.

`operation-accuracy/source-targets.npz` contains all world operation `points`;
optional `local_point_file` supplies `[T,2,K,3]` robot-local offsets. The v3 solver
constrains the first declared operation point per hand; all points are retained
for export/evaluation. Robot-local offsets are not scaled with source geometry.
Optional `target_rotation_zup[T,2,3,3]` supplies explicitly calibrated TCP frames.

## Outputs and integrity

The package requires a fresh output slot; a failed attempt is retained, not
overwritten. Raw `prime_original.npz`, `native_prior.npz` and
`source_prior_fullbody/*original.npz` have separate immutable SHA receipts.
`initial_qpos.npz` is the geometry initialization, distinct from the raw prior.
`gpu_qpos.npz` is final native motion and `motion.npz` is its exact named-joint
Grid export. `solver_targets.npz` retains actual objectives and GPU FK caches;
`source-targets.npz` retains all transformed operation targets.

`report.json` contains actual stopped status, provenance, model hashes, geometry
coverage and timing. `trajectory_scale` is the string `preserve|match_robot`;
`scale_transform={scale,anchor_xyz_m,world_up:"z"}` maps this request's source to
output. Historical source scaling remains separately under `morphology_scale`.
`VALID_STOPPED` does not establish nonlinear KKT or global convergence.

CPU handles licensed-source adaptation, local URDF/mesh compilation and sampled
geometry preparation, and native export. Neural encode/decode, native projection,
and resident FP64 refinement run on the selected GPU. Data are staged to the GPU
per solve, with small synchronization/diagnostic transfers between iterations;
this is not a zero-transfer implementation. Source encoding and target decode
can be deployed at separate trust boundaries in a future service, but this CLI
runs both locally. It never requires an upload of private geometry.

## Migration evidence

`PROVENANCE.json` records original module hashes. Tests cover MuJoCo-vs-Torch
kinematics/Jacobians, block solve vs dense solve, fixed/zero-row IPM behavior,
source all-body visibility, clock/integrity checks, similarity and physical
metadata resolution. A real 300-frame G1 migration test reproduced the original
v3 codes, raw/native priors, initialization, final qpos and solver targets with
maximum numeric difference **0.0** on the same workstation. All seven existing
robot metadata mappings were independently reconstructed identically. This is a
port/parity check, not a new quality benchmark or a cross-GPU throughput claim.

A separate real native SMPL-H 16-frame smoke passed source adaptation, the
workbench JSON/subprocess bridge, full refinement, exact native export and
artifact validation; no X conversion or source-body re-decoding was performed.
