# Supplied human motion and native SMPL-H reconstruction

The source adapter does not change the GPU refiner's mathematics. It normalizes
source evidence before the common body22 decoder and skeleton/surface/wrist
solver. An explicit supplied body never falls back to EgoAllo after a validation
error. Missing body data uses the existing native SMPL-H reconstruction from an
actual head pose and both wrist poses. Hands alone cannot invent a head.

## Interfaces

- `source_route(manifest)`: supplied body or three-point native-H route.
- `validate_three_point_observations(work)`: checks head+two wrists and 30 Hz clock.
- `prepare_body_source(work, output_dir, model_paths={family: local_model_npz})`
  / `prepare_source(...)`: normalize supplied input or completed native-H work.
  It never invokes EgoAllo itself. Output must be a fresh directory.
- `prepare_supplied_source(spec, output_dir, model_path=trusted_local_path,
  observed_work=None)`: direct parameter/FK/LBS or native decoded-body adapter.
- `describe_source(spec_or_work, model_paths=...)`: read-only neutral height and
  bone lengths; no neural inference, animated LBS, solver or target robot.

The endpoint may configure `ARENA_SMPL_MODEL`, `ARENA_SMPLH_MODEL` and
`ARENA_SMPLX_MODEL`. Uploads cannot specify an arbitrary server model/decoder
path, pickle, network URL or command. Model files/weights are not exported.
The local model must match the declared family and source betas. When supplied
fitted joints exist, the current model's FK is checked against them.

## Input descriptor

A recording may have `motion_source`:

```json
{
  "format": "smpl_parameters",
  "family": "smplx",
  "path": "motion.npz",
  "world_up": "y",
  "units": "m",
  "evidence": "fitted",
  "floor_z_m": 0.0,
  "pose_convention": "absolute_axis_angle"
}
```

`path` is package-relative at upload and resolved only within the package.
Optional `objects_path` is a package-local normalized NPZ containing original
object transforms and embedded local meshes; it is not a remotely fetched asset.
The scalar floor is the same physical vertical plane expressed in either input
world convention. It is never refit from target motion.

Parameter NPZ: `poses[T,24|52|55,3]` (or flattened), `trans[T,3]`/`transl`, fixed
`betas`, and original timestamps or explicit frame time. All original pose
channels are decoded. SMPL-H/X MCP topology defines palm semantics. SMPL's
24-joint body has no MCPs and requires explicit `wrist_axes_local[2,3,3]`, with
columns distal/transverse/normal. Nonzero expression or PCA/mean-offset hand
parameters need an explicit upstream decoder; the adapter rejects them rather
than silently dropping them.

Decoded NPZ: `joints[T,J,3]`, `global_rotation[T,J,3,3]`, `vertices[T,V,3]`,
`faces`, `betas`, `timestamps_s`; rotations map native model Y-up local axes to
the declared world. This accepts the actual native `smplh_motion.npz` exporter.
An H pelvis `root_translation` is not reinterpreted as SMPL `transl`.

Only nominal 30 Hz with original integer frame selection is supported. The
HiPhi 90.000900009 Hz source can use stride 3; exact timestamps and source frame
indices are retained, including its 10 ppm difference from nominal 30 Hz. Other
rates require an explicit separately recorded conversion, not an fps relabel.

## Output contract

`native_source.npz` contains body22 `names`, `parents`, `rest_offsets_cm`,
`full_body_position_yup_m`, `full_body_rotation_yup`, exact `timestamps_s`,
`source_frame_indices`, `exact_fps`, six positions/rotations, wrist task targets,
palm axes, actual source sole heights and source floor. The canonical local and
world axes are Y-up. With `C=[[0,1,0],[0,0,1],[1,0,0]]`, points convert to native
Z-up by `p_z=p_y@C`, global canonical-frame rotations by `C.T@R_y@C`; native
physical joint frames retain model-local axes and use `C.T@R_y`.

`source_mesh.npz` contains source vertices in world Z-up metres, original
triangle topology, the same timestamps and frame indices. Native H stays H and
external X stays X in provenance; body22 is a solver contract, not a renamed
SMPL-H parameter file. Original pose files remain immutable outside this bundle.
No X-to-H fitting or H-to-X transfer is performed.

For reconstruction or uploaded data with actual observations, `observed_work`
preserves `observations.npz`, `source_hands.npz` and
`operation-accuracy/source-targets.npz` byte-for-byte. The body reference still
contains reconstructed joints, while wrist position and semantic-axis task
references use the original observations; the six-point task view's wrist
slots use those same original wrist targets. Body22 positions remain the actual
source body geometry. These must share an exact clock.
`observed_source_sha256` and `task_reference_evidence` record that distinction.
Without independent observations (HiPhi example), task wrists come from the
supplied fitted body and are labelled accordingly, not as raw mocap truth.

Cache keys should include source/model/observation/hand/operation hashes,
source descriptor, scale policy and adapter source SHA. A complete normalized
bundle's own artifact hashes are in its manifest. Per-robot morphology scaling
belongs to the backend; this adapter always exports the original source space.

## Motion-only import and sample deployment

A supplied-body `arenalabs.recording.v1` package may omit video/camera and
observations when `end_effector=none`. It is registered as a real motion-only
recording, not a fabricated RGB/CV result. Actual hand/gripper components still
require their observed data. Raw archives and model assets remain outside the
served directory.

The ten HiPhi manifest entries in `examples/motion_sources/hiphi10.json` refer
to operator-staged data outside Git. `tools/register_motion_sources.py` checks
source hashes and resolves package-local paths before updating an explicitly
selected test registry/catalog. See that directory's README for source/license
scope. Registration neither downloads nor grants rights to the underlying data.
