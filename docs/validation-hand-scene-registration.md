# Source-hand global scene registration — 2026-10-19

Robot playback and original source observations are unchanged. Estimated human
hand/arm surfaces, manipulated objects, 3D boxes and the background use one fixed
right-handed similarity transform for the entire clip. This is geometry
registration, not retargeting, mesh completion or a hand-object contact solver.
The UI source human checkbox changes visibility only.

CPU validation reused immutable MapAnything predictions and SAM masks:

| Cached clip | Correspondences | Held-out samples | Before median | After median | After p90 | Uniform scale |
|---|---:|---:|---:|---:|---:|---:|
| load_dispense_ice/0 | 1,163 | 243 | 240.1 mm | 50.9 mm | 126.9 mm | 0.5083 |
| add_remove_lid/1 | 717 | 156 | 60.6 mm | 27.3 mm | 75.4 mm | 0.7922 |

These are observed-landmark correspondence residuals on temporally held-out
frames, not measured contact or object pose accuracy. Similarity fitting can
change the estimated scene's uniform scale; rigid fitting is preferred when
validation is within 5% of similarity fitting. The sizable residuals demonstrate
that one global transform does not repair local depth shape errors. Neither
sequence is claimed to be fully spatially calibrated.

The robot GLB and playback hashes matched the original downloaded packages.
Five sampled geometry frames per clip verified separate human/scene layers,
shared registration, original timestamps and object-box placement. Maximum box
versus canonical point percentile difference was 0.237 mm (ice) / 0.414 mm (lid),
within native depth quantization. Optical camera poses remain in their original
source world. Masks and model-native depth are not edited. Existing FOV and
video contact estimates were retained; source-fingertip proximity was recomputed
on the registered scene using the existing CPU path.

Registered review geometry export took 43.7 s for 1,057 source frames and 10.4 s
for 266 source frames, including RGB-D encoding; no reconstruction, segmentation
or robot inference reran. The UI streamed both layers on the same observation
clock and was checked with the source layer shown/hidden and time seeking.

Four registration/export tests plus twelve RGB-D tests passed, including a
translation/scale regression ensuring boxes and points share a space while the
optical camera is preserved. Seventeen browser-module tests and the frontend
production build passed. UMI/real gripper-origin proxies are not invented as
human landmarks; absent observed correspondences are explicitly reported, with
an identity registration. Displaying the observed human cloud still works.
