# MapAnything RGB-D rollout validation, 2026-10-19

Default geometry is camera-conditioned `facebook/map-anything-apache`.
SAM/Qwen masks, SMPL-X and six-point robot motion are independent and reused.
No contact inference, meshes, splats or historical fusion were run in this study.
DA3 remains an explicit comparison backend; old caches remain readable.

## Actual GPU runs

Each row used one RTX A6000, 32-view windows and 8-view overlap, local unchanged
weights, and synchronized optical-camera anchors. Times include model loading,
inference and native-output serialization; exclude preparation, SAM, robot
generation and FOV rendering. UMI/real camera calibration is estimated, not measured.

| Source | Source frames at 30 Hz | Depth observations at 15 Hz | Depth worker seconds | Canonical RGB-D bytes | Lossless native prior bytes |
|---|---:|---:|---:|---:|---:|
| EgoDex | 266 | 133 | 99.43 | 15,307,852 | 133,290,001 |
| Real robot | 550 | 275 | 157.32 | 33,589,321 | 235,341,683 |
| Nymeria | 450 | 225 | 135.24 | 35,750,039 | 208,812,058 |
| UMI | 96 | 48 | 69.90 | 7,628,359 | 37,433,682 |

The native predictions were originally saved as NPZ, then losslessly converted
to Blosc/Zstd bitshuffled per-window Zarr. Every saved native field and all
681 selected depth/confidence observations compared exactly. Original NPZ and
failed experiment attempts remain separate, immutable artifacts. New workers
write the native Zarr directly; these timings do not establish its speedup.
There is no second assembled float32 depth/confidence sequence.

Canonical RGB-D is the working/delivery representation: uint16 optical depth in
millimetres, camera Parquet, instance RLE and aligned RGB proxy. Native priors
are an additional audit cost, not included in the canonical size column. Do not
extrapolate only canonical bytes and call that total permanent storage.

## Integration checks

- H2 + Wuji FOV was actually rerendered using the new depth and original robot
  playback: 266 frames, 14 arm meshes and 52 hand meshes. Input motion was not
  recomputed; source SHA256 and clock were unchanged. Other robot FOV caches
  were not regenerated in this study.
- The new browser reconstructs full native valid depth, retains only 12 frames,
  uses bounded lookahead and excludes mismatched timestamps. RGB preview JPEG
  reduces transfer bytes; depth and instance labels remain lossless.
- Four source types passed the actual geometry HTTP endpoint. An exported
  LeRobot dataset was opened by the official reader at frames 0, 133 and 265,
  including 77 joint states, original RGB and inpaint RGB. The portable reader
  reconstructed 4096 points per requested observation with explicit source IDs.
- Cloud tests: 32 passed, 2 platform-specific skips. Browser unit tests: 12 passed.
  TypeScript and production frontend build passed. Local Windows upload testing
  needs FFmpeg on PATH; the cloud upload test passed with installed FFmpeg.

These tests establish integration, serialization and clock preservation, not
ground-truth depth accuracy or physical feasibility. A denser cloud is not
evidence that its geometry is more accurate. This rollout does not imply all
library recordings have already been recomputed with MapAnything.

See [the storage and calibration contract](point-cloud-direction.md).
