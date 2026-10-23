# Recording uploads and complete dataset downloads

The workbench uses one generation path for uploaded recordings and presets:
observed head + hands/end effectors → windowed EgoAllo/native SMPL-H → six-point
AlphaMotion (pelvis, head, both hands, both feet) → original robot refinement.
Depth inference, Qwen/SAM 3.1, FOV compositing and contact estimation remain independent
saved stages. Downloading an episode runs no inference.

## Upload contract

Use **Upload data** in Manipulation. Upload a ZIP containing `recording.json`,
`observations.npz` and `original.mp4`. The downloadable `pack_recording.py`
creates this package from an existing NPZ and MP4. Alternatively use
`--egodex-hdf5 recording.hdf5` instead of `--observations` to convert native
EgoDex head-camera and 21-point hands (requires h5py). It requires Python, NumPy
and FFprobe; it does not estimate missing input trajectories.

```sh
python pack_recording.py --observations observations.npz --video original.mp4 \
  --floor-z 0 --end-effector wuji --title "My manipulation" --output recording.zip
```

All arrays use a shared right-handed, **Z-up world in metres**, sampled at
**30 Hz**. The original source clock may have a nonzero origin; its intervals
must be 1/30 s. Only the time origin is normalized. No smoothing or pose fitting
happens during import. Camera poses map optical coordinates (right, down,
forward) to the world. Intrinsics refer to the original, undistorted MP4 size.

| NPZ key | Shape | Meaning |
|---|---|---|
| `timestamps_s` | N | Shared source clock |
| `camera_to_world` | N × 4 × 4 | Head camera poses |
| `intrinsic` | 3 × 3 | Pinhole camera matrix |
| `wrist_position` | N × 2 × 3 | Left/right observed wrist or declared fixed housing reference |
| `ee_rotation` | N × 2 × 3 × 3 | Local EE axes expressed in world; +Z forward, +X palm normal |
| `hand_points` | N × 2 × 21 × 3 | Required for Wuji: wrist, then thumb/index/middle/ring/little, four landmarks each |
| `hand_confidence` | N × 2 × 21 | Optional observed-landmark confidence |
| `gripper_width_m` | N × 2 | Required for grippers, opening in metres |

The manifest declares `schema: arenalabs.recording.v1`, `fps: 30`, `units: m`,
`world_up: z`, `floor_z_m`, `image_size: [width,height]`, `video`,
`observations` and `end_effector` (`wuji`, `none`, `robotiq-2f-140`).
`video_start_seconds` defaults to zero. Only the head camera and bilateral
end-effector observations enter body reconstruction; source robot joints,
pelvis and feet are excluded. A bare video is not a calibrated motion input.

Input recordings from EgoDex, Nymeria, UMI or a robot can be packaged after
conversion to this explicit contract; the upload endpoint does not guess
vendor schemas, lens models, world units or camera extrinsics. Presets retain
their existing native adapters. Archive limit: 2 GiB compressed, 8 GiB expanded.

Uploads use a browser workspace cookie. They are absent from other browsers'
libraries and their assets require that cookie. This is browser-workspace
isolation, not an account or collaboration system; clearing the cookie loses
browser access. Originals remain immutable in private server storage.

## Download contract

Select a generated robot and use **Download dataset**. The ZIP is a standalone
LeRobot v3.0 dataset, tested with the official `lerobot==0.4.3` reader:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("arenalabs/my-export", root="/path/to/unpacked-dataset")
sample = dataset[100]
```

`data/` contains frame-indexed Parquet; `videos/` contains original, inpainted
and available wrist FOVs resampled to the common 30 Hz clock; `meta/` contains
features, units, tasks, episodes, statistics and video offsets. The native
source video is also retained among the extension assets.

`observation.state` is generated joint position, not measured telemetry.
`action[t]` is the next-frame kinematic joint-position reference; the final
frame repeats. These arrays are not actuator commands or evidence of task
success. Mixed hinge/slide joint units remain explicit in the extension manifest.

`extensions/arenalabs/` retains the complete playback, robot asset, point cloud,
semantic instance tracks and boxes, contact/proximity JSON (including nulls,
evidence and native sampling rate), semantic labels, camera calibration,
native depth/confidence/camera predictions, lossless SAM masks and saved human motion. Model
templates, licensed checkpoints, credentials and unrelated recordings are
excluded by an explicit allowlist. Missing stages are declared as missing.

`frame_map.parquet` links 30 Hz frames to actual cloud observations (normally
15 Hz); no cloud or contact measurements are fabricated between samples.
Use the included `read_extensions.py` to load legacy cloud buffers or reconstruct
points from the new compressed RGB-D bundle. `cloud_at(dataset, frame_index,
maximum_points=4096)` samples on demand; no neural decoder is needed. New geometry
uses uint16 mm Zarr, aligned RGB proxy video, RLE labels and camera Parquet.
Original float32 model depth and calibration predictions remain in `raw/`.
Masks
retain their prepared image pixel space and source frame indices; camera and
display transforms remain in `scene.json`. A 3D box bounds the visible estimated
surface, not an inferred complete physical asset. Contact estimates are not
measured pressure.

Every packaged file has a SHA256 entry in `checksums.json`. Exports are immutable
CPU jobs; failures retain logs and do not replace earlier outputs. Nothing is
automatically uploaded to Hugging Face.

## Source human cloud and global scene registration

New scene exports retain observed hand/arm pixels as a separate optional
`human_depth_mm` array in the same RGB-D bundle. Scene depth continues to exclude
human/capture hardware. `cloud_at(dataset, frame, source_human=True)` returns
only that reference layer. Default point-cloud and proximity consumers continue
to see only the scene; source-hand points never become manipulable objects.

A single clip-wide right-handed rigid/similarity registration maps raw estimated
geometry to observed source hand/arm trajectories. The same transform applies
to background, object surfaces, 3D boxes and source human points. Robot motion,
source trajectories, camera measurements, model-native depth and masks remain
unchanged. `geometry.hand_alignment` reports the matrix, uniform scale, anchor
provenance, temporal validation split and before/after residuals. Geometry index
`display_from_source` includes this transform; apply it exactly once. Physical
camera Parquet poses remain separate. The browser packet's `depth_to_display`
is the effective affine, potentially including uniform scale, not an optical
extrinsic calibration.

The source human layer's Display checkbox controls visibility only. Registration
is computed during geometry export and retained even when this layer is hidden.
No extra reconstruction or motion solve is required for a visibility change.
The solve uses confidence-filtered projected observed landmarks on hand/arm
masks, robust 3D correspondences, and held-out frames. It does not guarantee
contact or correct nonrigid/local depth errors. When no reliable observed
correspondences exist, registration is explicitly unavailable/insufficient,
with an identity transform; the observed human layer can still be viewed.
UMI gripper/device origins are not silently treated as calibrated human landmarks.
Raw alignment correspondences and audit JSON are included in the lossless
extension when available. Existing suitable depth/mask caches are upgraded by
CPU export, without rerunning MapAnything, SAM or robot motion.

## Isaac Sim replay

New exports (`arenalabs-lerobot-v3.0-3-isaac-replay`) include
`extensions/arenalabs/replay/robot.usda`, a self-contained 30 Hz USD animation.
Open it in Isaac Sim and press Play, or use the included `isaac_replay.py`
with Isaac Sim 5.1's `python.sh`. The replay folder README gives GUI and
headless validation commands. Exporting it is CPU-only and does not rerun any
motion model. Existing immutable ZIPs are not rewritten; download a new export.

The USD embeds all exported robot meshes, mounted hands/grippers and full
world motion. It is **kinematic visual replay**, without a PhysX articulation,
actuator controller or claim of dynamic stability. No URDF import, private
model path, model weights or network asset is required for this replay.
For control/simulation, a matching physical robot description, explicit named
DOF mapping, floating-base convention and a controller remain necessary.

The source playback uses metres, Y-up and global mesh orientations in xyzw.
The USD applies one right-handed basis rotation `(x,y,z) -> (x,-z,y)` to Z-up;
USD quaternion tuples use wxyz. World geometry poses must never be interpreted
as joint-local transforms. `replay.json` records this contract. LeRobot's named
joint references include passive/coupled/hand joints and are **not qpos**;
`action[t]` also refers to frame t+1. Use the USD for exact visual replay.

The included Isaac validator compares every mesh transform at every frame to
the original exported playback, then renders forward/backward timeline seeks.
It fails if translation or rotation-matrix error exceeds 1e-6, geometry is
missing, external USD layers are needed, or rendered images are empty. Numeric
verification is separate from contact/dynamics validation.

## Code boundaries

- `dataset_io/ingest.py`: archive validation, coordinate/clock checks, adapter preparation.
- `dataset_io/storage.py`: path boundaries, registry and workspace ownership.
- `dataset_io/routes.py`: streaming HTTP import, library and asynchronous export endpoints.
- `dataset_io/export.py`: LeRobot writer and lossless modality packaging.
- `dataset_io/worker.py`: CPU export process.
- `dataset_io/replay.py`: self-contained USD geometry and animation writer.
- `dataset_io/isaac_replay.py`: package-only Isaac player and independent transform/render checks.
- `web/src/manipulation4d/dataset-io.ts`: upload and download dialogs.

The generation scheduler and solvers are shared with presets. The server needs
the existing inference runtimes plus PyArrow/Pandas in `ARENA_DATASET_PYTHON`
(defaults to the workbench runtime). See the main README for private weights,
robot assets and server configuration prerequisites.

Official format reference: https://huggingface.co/docs/lerobot/lerobot-dataset-v3
