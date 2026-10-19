# Current-frame geometry after retiring scene assets

Decision, 2026-10-19: revert the isolated scene-asset experiment. The new default
depth backend is camera-conditioned MapAnything with current-frame SAM 3.1
segmentation. DA3 remains an explicit comparison backend. SMPL-X, six-point
AlphaMotion, refinement, object masks/visible-surface boxes, FOV compositing
and ordinary dataset exports retain their existing behavior. No static 3DGS,
object mesh generation, FoundationPose tracking or historical-cloud fusion.
The rejected implementation and immutable experiment outputs remain recoverable
through Git history and the separate experiment directory.

## Depth-only backend

`scene_inference.py` dispatches to `mapanything_depth.py` by default. It uses
32-view windows with 8-view overlap and one synchronized optical-camera gauge.
For overlapping predictions it selects the result with the most temporal
context; it does not average moving objects or alter source task trajectories.
Native depth, confidence/validity and predicted camera parameters are saved in
immutable per-window float32 Zarr stores before known optical anchors are applied;
each native store has a deterministic content-tree SHA256 receipt.
No derived XYZ sequence is saved. A small Zarr camera/validity index points into
the original float32 window predictions, avoiding a second full depth/confidence
copy. The lazy `depth_store.load()` contract retains at most two native windows
per accessed field and also reads old assembled Zarr and DA3 NPZ caches.
SAM remains a downstream mask provider and runs in parallel with depth.

Provision MapAnything and local weights separately. Set
`ARENA_MAPANYTHING_PYTHON`, `ARENA_MAPANYTHING_WEIGHTS` and make the upstream
MapAnything package importable in that interpreter. Install this repository's
`geometry` extra in the scene, dataset and API environments. The default model
is `facebook/map-anything-apache`; inference loads only a local weights path.
`ARENA_DEPTH_BACKEND=da3` selects the old comparison branch. Window size and
overlap have explicit environment overrides and enter scene cache identity.
Existing DA3 clouds remain readable; they are not marked as MapAnything caches.
Body/robot motion caches remain independent of this geometry policy change.

Without any optical poses only a single bounded window is supported; long
videos fail explicitly rather than independently resetting camera scale/gauge.
UMI/real-robot sources reuse available estimated camera intrinsics or bootstrap
them from MapAnything using synchronized source poses, undo preprocessing,
then rerun with those estimates. Estimated calibration remains labelled.

Current DA3 inference uses process_res=504. Its cloud exporter keeps every
second row and every second column, retaining roughly one quarter of available
depth pixels. Compare native depth geometry separately from rendering density.
Upsampling or enlarging point sprites does not recover measured surface detail.

Use identical source frames, coordinate conventions and declared calibration;
preserve the original trajectories. Compare object edges, foreground/background
depth bleeding, table shape, temporal flicker, source alignment, latency and
GPU memory. Inspect native validity masks separately: different rejection masks
can otherwise make one method appear cleaner without better depth.

Known head-camera trajectories should condition the depth backend when its API
supports calibrated inputs. A head/device pose first needs the fixed optical
camera extrinsic, correct source timestamps and pixel-space intrinsics. The
existing DA3 branch already passes available calibrated camera poses; this is
not a newly added capability. Pose knowledge enables short-window static
reprojection/depth-consistency checks, but does not itself correct timestamp
offsets or turn learned depth into ground truth. Use SAM dynamic exclusions and
occlusion tests; never force moving objects into the static camera-motion model.
Any temporal filtering stays separate from immutable predictions and does not
require retaining or displaying historical point clouds.

A cached diagnostic on the same EgoDex clip used DA3 504x280 and MapAnything
518x294. Source frame 132 retained 122,377 and 139,023 native valid points,
respectively, with their own masks. This is a density/visualization check, not
ground-truth geometric validation or a verified model-quality improvement.

## Canonical representation and on-demand points

- Keep source RGB once, in the original video encoding when possible. The
  geometry bundle also has a small, encoded RGB proxy in exactly the depth
  pixel space for efficient random-access colours. It is a derived video,
  not reconstructed novel-view imagery.
- Store camera optical Z-depth as chunked Zarr with Zstd/Blosc compression.
  Retain immutable model priors separately as required by repository policy.
  Float32 metres preserves model depth values; an optional uint16 millimetre
  delivery representation adds at most about 0.5 mm rounding error within range.
  Zero means invalid; values above 65.535 metres must never wrap or clip silently.
- Save intrinsics, optical camera-to-world poses, timestamps, source frame IDs,
  pixel preprocessing transforms, units and axis conventions explicitly.
- Keep SAM instance masks as RLE, semantic metadata and trajectories in the
  existing export contract; preserve confidence/validity when available.
- Reconstruct XYZ deterministically from depth and calibration on demand. Use
  the aligned source RGB for colour and SAM masks for labels. No inference
  network is needed to decode this representation.
- Sample client-requested 1024/4096/8192 points only for delivery. A browser
  cloud buffer is a disposable cache, not the authoritative permanent copy.

The public bundle contains `depth.zarr/depth_mm`, `cameras.parquet`,
`masks.rle.jsonl`, `rgb.mp4` and `index.json`. Invalid, human and hardware depth
is zero in this derivative; original model predictions and SAM masks remain
private audit artifacts. Pixels outside uint16 range are invalid, never wrapped.
`/api/4d/geometry/frame` delivers a gzip-compressed RGB-D packet after checking
workspace ownership. The browser backprojects full native valid depth and
retains only 12 frames in memory, prefetching a bounded lookahead to cover network
latency; no permanent `cloud.bin` is generated.
Browser requests use a JPEG colour preview within the packet; depth and labels
remain lossless. This preview does not replace the original or proxy RGB video.
The API keeps at most 4 proxy decoders with 24 decoded RGB frames each, avoiding
reopening a video codec for every observation. Neither cache grows with duration.
LeRobot exports include the same bundle and a portable `cloud_at()` reader;
`maximum_points=4096` samples points on request. Geometry retains 15 Hz source
IDs while robot states and original/FOV videos retain the 30 Hz episode clock.

Do not promise a compression ratio before measuring representative episodes.
At 504x280, 15 Hz, uint16 depth alone is 15.24 decimal GB/hour before lossless
compression (30.48 GB/hour for float32). At 960x540 it is 55.99 GB/hour for
uint16. These exclude RGB, masks, confidence and immutable priors; 30 Hz doubles
the depth budget. Native model resolution and source clocks must be disclosed.

References:
- https://github.com/facebookresearch/map-anything
- https://github.com/microsoft/MoGe
- https://github.com/ByteDance-Seed/Depth-Anything-3
