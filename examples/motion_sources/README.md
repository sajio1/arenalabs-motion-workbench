# Supplied motion examples

This directory contains manifests and import instructions only. Actual HiPhi,
GRAIL, videos, meshes, licensed body models and weights stay outside Git.
The operator explicitly stages authorized data on the test endpoint, then runs
`tools/register_motion_sources.py --manifest ... --data-root ... --registry ...`.
Nothing downloads automatically. Registration does not expose raw source files.

`hiphi10.json` selects ten real source windows using source posture and operation
coverage, not target-robot solver quality. Each contains 300 frames selected at
stride 3 from the original clock (30.000300003 Hz, nominal 30 Hz); no interpolation
or speed change. These are local `gpu_convert v2b_cpubeta` conversions using the
CPU converter's betas, derived from official `hiphi2smplx@65e5131`. They are fitted
SMPL-X, not dataset-released native SMPL-H, not independently rechecked BVH ground
truth. All 55 pose channels are retained; face/hand channels are zero and are not
observed finger actions. No original egocentric RGB is available in these examples.

HiPhi upstream: Noitom Robotics, HiPHI, ModalityNet Open Research License 1.0,
2026-08-13: https://huggingface.co/datasets/noitomrobotics/HiPHI/blob/main/LICENSE.md
The upstream public license does not grant public raw-data redistribution or
commercial product evaluation. For this test, the operator explicitly confirmed
separate authorization in the 2026-10-22 request; that confirmation is not a
license grant to repository users. Obtain your own applicable permission before
staging or using data. Do not infer permission from this manifest or from a
private repository setting. The actual data is deliberately absent from Git.

GRAIL source: NVIDIA PhysicalAI Robotics Locomanipulation GRAIL,
https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL
Its motion trajectories/reconstructions use CC BY-NC 4.0; object assets and model
weights can have separate upstream terms. This repository does not redistribute
them. A supplied full SMPL-X reconstruction uses the same direct adapter; a robot
recording without a supplied human body must provide real head+two wrist poses
for native SMPL-H reconstruction. A 24/25 Hz GRAIL source needs an explicit,
separately recorded time conversion before the 30 Hz runner; do not relabel fps.
Untrusted pickle imports are not supported by the upload endpoint.
