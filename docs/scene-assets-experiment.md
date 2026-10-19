# MapAnything scene-asset workflow

This is a separate experimental entry point. The hosted page, DA3 comparison
baseline, existing motion policies and caches are unchanged. Compute runs on
cloud workers; nothing starts when the package is imported.

## Processing graph

```text
Immutable RGB / source clock / optional calibrated optical cameras / task poses
    |-- existing windowed EgoAllo -> SMPL-X -> six-point AlphaMotion -> refiner
    |     (all original palm/gripper targets stay unchanged)
    `-- existing Qwen discovery -> SAM 3.1 object / arm / hardware masks
          -> bounded MapAnything multi-view chunks
          -> static-only overlap registration + verified non-overlap loops
          -> sparse global Sim3 pose graph, or fixed supplied camera anchors
          -> disk-backed depth / camera / mask scratch store
             |-- adaptive gsplat static training -> official SPZ encoder
             `-- original-resolution object masks + MapAnything pointmaps
                   -> SAM 3D Objects complete mesh hypotheses
                   -> FoundationPose pretrained registration / tracking
                   -> compact GLBs + nullable source-world 6DoF trajectories
          -> one robot-independent canonical asset package
          -> deferred customer-robot / camera rendering
          -> complete LeRobot v3 motion + RGB episode, depth / points in Zarr
```

SAM completion is a dependency of static registration and fitting. Human-motion
generation remains independent. GS fitting and learned object generation can
run on separate GPU leases once geometry is ready. No second cloud machine or
production service is provisioned by this entry point. Finger contact is absent.

The old TSDF / rigid ICP object prototype has been removed from this implementation.
Its immutable experimental results and rejection receipt remain separate evidence;
they are not used as a substitute for the requested mesh-generation branch.

## Inputs and coordinate contract

Input is the existing prepared scene-job directory: `manifest.json`,
`frames.npz`, `frames/*.jpg`, `prompts.json`, `masks/*-masks.npz`, optional
`cameras.npz` / `camera-intrinsics.npz`. Source frame IDs must agree with SAM.
The same policy applies across datasets, robots, actions and frame intervals.

Only optical OpenCV RDF camera-to-world poses belong in `cameras.npz`.
Head/device poses need a declared fixed optical-camera extrinsic first. UMI
gripper trajectories are not camera poses. A supplied estimated calibration
remains an estimate; presence of a file is not proof of measured calibration.
Without source-world optical calibration, reconstruction coordinates are estimated
and cannot be presented as a measured alignment with robot task trajectories.

MapAnything can adjust supplied inputs and uses a reference-camera gauge. Its
raw outputs are saved per chunk without replacement. For supplied optical
calibration, final depth is backprojected with unchanged input K / camera poses.
Neither ICP nor a pose graph moves source task observations or those anchors.

For estimated poses, static common-frame 3D correspondences give overlap edges.
Non-overlap candidates require distinctive static image matches, a robust 3D
fit and spatially distributed inliers. Image similarity alone is insufficient.
The global sparse solver fixes node zero and corrects chunk Sim3 transforms.
A textureless short clip may have zero verified loop edges; the audit reports
this rather than inventing a closure. Disconnected overlap fails explicitly.

Upstream MapAnything has no arbitrary external SAM-mask input key. This adapter
uses the masks in registration, verified loop matching, static initialization,
photometric/SSIM and depth fitting. Objects are excluded conservatively from the
static scene. It does not claim dynamic pixels never influence backbone attention.
The same pointmap is also a supported direct input to SAM 3D Objects; its camera
convention is converted with the upstream helper, not an assumed axis flip.

## Long-video memory and identity

The model sees one fixed-size overlapping chunk. Raw priors are immutable NPZs.
Scratch arrays are frame-chunked Zarr; older NPZ members are extracted once to
NPY memmaps. Registration maps at most two chunks, GS uses a bounded seed
reservoir and transfers one training frame at a time to CUDA. Dense XYZ is
computed from depth/K/pose as needed, not retained as a sequence in memory.
Sparse Jacobians avoid a dense, quadratic-memory pose graph solve.

SAM temporal IDs are retained before cross-fragment matching. Fragments visible
together anywhere in the recording cannot be merged into one object. Ambiguous
matches remain separate. This prevents spawning a new asset every frame, but
does not prove perfect physical identity when SAM itself switches IDs.

Object seed frames are selected by reliable visible mask area, without manual
frame picking. Original-resolution RGB/SAM observations are paired through the
exact preprocessing pixel affine with MapAnything depth. SAM 3D's raw full mesh,
layout scale/rotation/translation and checksum stay outside the compact asset.
A shared official visibility-aware mesh postprocessor simplifies surfaces,
removes invisible components and repairs small boundaries; raw priors survive.
Plain 20,000-face decimation introduced holes in closed meshes and was rejected.
The model-generated backside is explicitly a hypothesis, not a sensor observation.

FoundationPose uses its pretrained scorer/refiner, with registration after long
gaps and tracking in both directions from the best observed seed. It never
borrows object motion from a wrist. Unobserved rows stay nullable. Surface-fit
diagnostics compare MapAnything mask points with the generated asset; these
are not ground-truth pose, grasp, contact or execution-success measurements.

## Permanent storage

| Asset | Format |
|---|---|
| One native evidence video | MP4, retained once per source |
| Poses, intrinsics, input observations, SMPL-X parameters | Zstandard Parquet with shape / dtype metadata |
| Native segmentation and separate geometry-resolution labels | COCO-style RLE JSONL + ordinary gzip |
| Trained static scene | Official SPZ |
| Object hypotheses | GLB, one per persistent track |
| Object 6DoF / visibility / diagnostics | Parquet, original source frame IDs |
| Provenance, transforms, checksums, statuses | JSON |

No per-frame cloud, dense SMPL mesh sequence, learned private decoder or model
weights belong in the permanent package. Derived SMPL vertices/faces/joints are
omitted only for a recognized parameter schema; original observed joints are
not removed. Raw model priors and failures remain separate research scratch
until their retention is resolved. Source data and model licenses still apply.

The target of a few GB per source hour needs measurements across actual scene
complexity, object count and source bitrate; it is not a guaranteed compression
ratio. A short scene's asset size cannot be multiplied linearly to infer an
hour with different objects or viewpoints.

## Deferred delivery

`render` consumes canonical assets, target-camera Parquet and, when requested,
the existing URDF ingestion/retarget output: URDF identity, native model GLB,
matching per-geometry playback, a declared robot-to-world transform and a
checksum-bound geometry/pose order receipt. Robot paths and task coordinates
are not recomputed or registered to the scene by the renderer.

GS expected Z-depth and object/robot mesh z-buffers are composed in one optical
camera. Adjacent observed 6DoF samples may be interpolated to the 30 Hz motion
clock. Null observations and out-of-range poses are never filled by extrapolation.
Surface-origin flags distinguish static GS, generated object and robot CAD.
The renderer supports changed camera height, but cannot certify unseen geometry.

With a complete native robot playback, `delivery_join` reuses the current
exporter's video normalization, statistics and LeRobot v3 table conventions.
The final episode has joint states, next-frame joint-position references,
original and rendered RGB, tasks / episode / statistics metadata. Clock and
video coverage must match exactly. Scene-only renders remain labeled fragments.

Depth is Zarr uint16 millimetres, zero invalid, above 65.535 m invalid rather
than overflow. Separate validity and instance maps accompany depth and sampled
clouds. Supported samples are 1024 / 4096 / 8192 points; padding never duplicates
points. XYZ float32 + RGB uint8 + instance uint16 + validity uint8 is 18 bytes
per slot: at 30 Hz, raw cloud budgets are 1.99 / 7.96 / 15.93 GB/hour; 4096
points at 15 Hz is 3.98 GB/hour. RGB, depth and labels are additional.
RGB/actions remain 30 Hz. Geometry defaults to 15 Hz with explicit Zarr
`source_frame` and `timestamp_s` arrays; `--geometry-fps 30` retains every frame.
Neither delivery readers nor training code should assume those clocks coincide.

Deliveries have 7–30 day expiry. `expire ROOT` lists owned expired immediate
children of a dedicated temporary-delivery root; `--delete` removes only those.
Canonical assets, symlinks and unmarked directories are protected. No cron job
or hosted route is silently installed.

## Provisioning and commands

Keep the geometry/GS environment separate from the SAM 3D / FoundationPose
environment. An explicit model-worker Docker recipe is provided in
`tools/scene-assets-models.Dockerfile`; do not install it into the live service.
The tested stack uses NumPy 1.26.4 and PyArrow 19.0.1. NumPy 2 / old Kaolin,
latest utils3d / old SAM imports, and old system pybind11 / Python 3.11 have
known API/ABI conflicts; source revisions are pinned in the recipe.

Acquire weights explicitly under their upstream licenses, outside Git:

- MapAnything Apache checkpoint `facebook/map-anything-apache`, revision
  `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`; source
  `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`.
- SAM 3D Objects `facebook/sam-3d-objects`, revision
  `2e73555018d2741ccd486e56c24fac41155a1dc6`; source
  `f91db411c50efee93d8db7aeb323885650f6f722`. HF access is gated.
- FoundationPose source `a1b694b83e633c2cb6115b9063d940a687759392`;
  official scorer/refiner weight folders `2024-01-11-20-02-45` and
  `2023-10-28-18-33-37` in that checkout's `weights/`.
- The upstream DINO code/weights cache must already be provisioned for offline
  model initialization. Model inference must not depend on an HF login token.

```bash
python -m arenalab_workbench.scene_assets run /prepared/job /isolated/job \
  --model /models/mapanything --converter /tools/spz/ply_to_spz \
  --sam-source /models/sam3d --sam-config /weights/sam3d/pipeline.yaml \
  --foundation-source /models/foundationpose \
  --object-python /explicit/model-worker-wrapper \
  --parameters /original/smplx_motion.npz /original/observations.npz

python -m arenalab_workbench.scene_assets render /isolated/job/assets \
  /target/cameras.parquet /temporary/delivery-001 --decoder /tools/spz/spz_to_ply \
  --robot-model /native/model.glb --playback /native/playback.json \
  --robot-to-world /native/declared-transform.json --urdf /target/robot.urdf \
  --robot-binding /native/binding.json --points 4096 --days 7

python -m arenalab_workbench.scene_assets expire /temporary
```

`tools/scene-object-worker.py` is the reference wrapper: make it executable,
set `ARENA_TORCH_CACHE_ROOT` and explicitly build the named Docker image. It
preserves GPU leases, mounts source/weights read-only and runs with no network.
The wrapper is an explicitly provisioned executable that runs the isolated model
interpreter. It must preserve the scheduler's `CUDA_VISIBLE_DEVICES`, map input
paths explicitly when using a container, and never download or change services.
`objects-track` retries failed tracking into a fresh directory without rerunning
SAM 3D. Stage receipts retain failed attempts rather than overwriting raw priors.

Numerical tests cover formats, masked registration/loops, sparse pose graph,
disk-backed storage, SAM identity continuity, camera anchors and nullable pose
interpolation. Actual adoption additionally requires visual and pose checks on
EgoDex, Nymeria, UMI and real-robot recordings, long revisits and customer camera
calibration. Preserve individual failures; do not present successful code
execution as production-quality reconstruction.

References: [MapAnything](https://github.com/facebookresearch/map-anything),
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects),
[FoundationPose](https://github.com/NVlabs/FoundationPose),
[gsplat](https://github.com/nerfstudio-project/gsplat),
[SPZ](https://github.com/nianticlabs/spz).

## Recorded 2026-10-19 pilot

See `validation/scene-assets-sam3d-20261019.json` for the current learned-object
run. The older `scene-assets-20261019.json` is the rejected TSDF prototype.

- EgoDex, 266 source frames / 8.87 seconds; existing SAM, human and H2/Wuji
  outputs reused. MapAnything has six overlapping windows. Recorded camera
  arrays and immutable human/observed-hand tensors round-trip exactly.
- Static scene: 7,000 adaptive-training steps, 52,238 Gaussians, 748,005-byte
  SPZ. Fifteen SAM-derived tracks produce meshes/poses; they are not guaranteed
  fifteen distinct physical objects. Official postprocessing produces fourteen
  watertight meshes. Watertightness does not certify shape or pose accuracy.
- Canonical package: 15,137,550 bytes, including the 5,358,928-byte original
  video. Complete temporary delivery: 47,053,968 bytes. No per-frame clouds
  occur in the permanent package. These are clip measurements, not hourly
  storage/cost guarantees.
- LeRobot 0.4.4 reads frames 0 / 133 / 265 with 77 joint states/actions and
  both original/rendered RGB. Geometry has 133 explicit samples at 15 Hz.
  Sampled XYZ reprojects to uint16 depth within the expected 0.5 mm rounding.
- Twenty numerical/storage tests pass on Linux. A separate pose-free run
  solves five overlap graph edges. This low-texture clip yields no verified
  non-overlap closure; long revisits are not yet validated.
- The experiment's hourly cleanup timer is active only for owned temporary
  deliveries, with filesystem writes restricted to that directory. Canonical
  assets and original evidence are outside that writable root. The model
  container is stopped; no production service was restarted.

The rendered comparison still exposes object identity, geometry and tracking
errors, including large errors on some partial masks. This is a functioning
experimental representation/export path, not a quality approval or a reason
to replace the main page yet. Nymeria, UMI, real-robot and long-video validation
remain separate work; the clean Docker build recipe has not been rebuilt in
this pilot (the explicitly provisioned isolated model environment was used).
