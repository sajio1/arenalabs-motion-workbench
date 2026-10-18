# Nymeria RGB finger experiment (2026-10-13)

Explicitly requested finger-only experiment, outside the production pipeline. Existing v7 body/wrist motion is immutable. No production defaults were changed.

Flow: RGB -> WiLoR-mini MANO pose -> externally reviewed observation mask -> local finger rotations in SMPL-X -> Wuji SDK retargeting -> visual playback. Model files and data stay outside Git.

## Result and limitations

This is NOT full observed finger recovery. The jacket clip has 7 accepted single-hand frames out of 900; the laundry clip has 10/900, some partially occluded. Most frames retain the existing EgoAllo finger prior. Masks are manual data-quality annotations, not a general automatic association method. Even accepted poses are monocular estimates without finger ground truth. No finger accuracy claim is made.

Full-image detections often belong to another person. Approximate wrist-guided crops and zoomed detections still produced cloth/background hallucinations; those trials were rejected. Diagnostic scripts and failed outputs remain in the external experiment workspace. A calibrated camera and robust wearer-hand tracking would be required for automation. No hidden interpolated frame is counted as observed.

Transfer blends reviewed rotations with the existing prior over nearby frames (sigma 2 frames), preserving the original body rotations, root and wrist targets. Fixed hand mounts are the pre-existing visual mounts, not hardware validation.

Validation: SMPL-X first 25 joints unchanged exactly; H2/A3 body mesh positions and rotations unchanged exactly; all Wuji angles finite and within URDF limits. Kinematic only, no collision/dynamics validation. Two new 720p 30Hz reviews remove the first 60 frames and preserve the accepted original videos separately.

## Running

These are experimental scripts with the existing cloud experiment layout, not a portable production command. Set ARENA_EGO_WORKDIR to the clip directory and ARENA_OUTPUTDIR for outputs. `select_reviewed_hands.py` consumes video-hands/reviewed-detections.json and raw-mano.npz. `apply_video_fingers.py` needs the original smplx_motion.npz and licensed SMPL-X model. Export also needs original native motion, robot metadata and Wuji assets. Use the isolated reconstruction/render environment; native robot export uses the existing SDK environment. Do not download models or publish data implicitly in normal pipeline runs.

Upstream: https://github.com/rolpotamias/WiLoR and https://github.com/warmshao/WiLoR-mini . Respect upstream model/data licenses. No weights included.
