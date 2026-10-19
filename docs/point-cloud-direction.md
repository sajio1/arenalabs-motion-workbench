# Current-frame geometry after retiring scene assets

Decision, 2026-10-19: revert the isolated scene-asset experiment. Production
remains the existing current-frame DA3 + SAM 3.1 pipeline. SMPL-X, six-point
AlphaMotion, refinement, object masks/visible-surface boxes, FOV compositing
and ordinary dataset exports retain their existing behavior. No static 3DGS,
object mesh generation, FoundationPose tracking or historical-cloud fusion.
The rejected implementation and immutable experiment outputs remain recoverable
through Git history and the separate experiment directory.

## Depth-only evaluation

Evaluate replacement depth backends without changing motion or perception.
MapAnything supports optional calibrated optical poses/intrinsics. MoGe's
current published implementation is another candidate for fine-detail metric
monocular geometry; temporal stability must be evaluated independently. Neither
is assumed to outperform DA3 on these recordings before direct comparison.

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

## Proposed permanent representation (not yet a storage migration)

- Keep source RGB once, in the original video encoding when possible.
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

Do not promise a compression ratio before measuring representative episodes.
At 504x280, 15 Hz, uint16 depth alone is 15.24 decimal GB/hour before lossless
compression (30.48 GB/hour for float32). At 960x540 it is 55.99 GB/hour for
uint16. These exclude RGB, masks, confidence and immutable priors; 30 Hz doubles
the depth budget. Native model resolution and source clocks must be disclosed.

References:
- https://github.com/facebookresearch/map-anything
- https://github.com/microsoft/MoGe
- https://github.com/ByteDance-Seed/Depth-Anything-3
