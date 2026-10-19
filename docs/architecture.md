# Production architecture

The production application is `arenalab_workbench`. Start it explicitly with
`arena-workbench --config /path/to/viewer.cloud.json`, or
`python -m arenalab_workbench`. Importing the entry point does not start a server.
The native `greenwich-pipelines` command remains a separate, supported batch
interface; its legacy Nymeria/Gento solvers are not the web human-prior route.

## Generation flow

1. Dataset adapters normalize the source camera/head and observed end effectors
   onto a shared 30 Hz clock. EgoDex keeps observed finger landmarks for Wuji;
   Nymeria requests no dexterous hand. UMI and real robot inputs keep the camera
   and two gripper housing poses, excluding recorded body/pelvis/leg state.
2. `body_prior`, `gripper_body`, `body_inference` and `run_egoallo_windowed`
   produce immutable human assets. EgoAllo keeps 30 sampling rounds, inner and
   post guidance, GPU JAX and its persistent compilation cache. Fixed windows
   overlap; padding is excluded from output. Raw priors retain their SHA256.
3. `window_robot_stream`, `retarget_robot` and `gripper_retarget` consume the
   six-point conditioning policy: head, pelvis, two hands and two feet.
   Robot refinement still follows original observed palm/gripper targets.
   A cached human asset can feed multiple robots without resampling the body.
4. `scene_job` runs DA3 depth and Qwen discovery/SAM segmentation as independent
   GPU branches. SAM masks project into DA3 points, retaining semantic labels,
   instances and 3D boxes. The viewer renders the current source frame only.
5. Robot FOV rendering/inpainting and optional contact estimates finish their
   own modalities. `progressive_job` publishes motion, then scene, then FOV and
   contact readiness; partial perception completion never completes a robot job.
6. `dataset_io` owns normalized uploads, workspace authorization, immutable
   exports and portable dataset readers. Ownership guards also protect files.

`full_job` and `gripper_job` orchestrate these stages. `gpu_scheduler` provides
cross-process GPU leases and resource locks; model workers run in isolated
interpreters. `scene_job` snapshots a complete importable worker package into
its private job, so qualified imports cannot accidentally use live modules.
The scene cache hashes its relevant workers and configuration, independently
of robot output. Viewer conversion fingerprints follow conversion dependencies
and policy files instead of hashing every web/API module in `src`.

## Module boundaries

| Module | Responsibility |
|---|---|
| `serve_workbench`, `workbench_api`, `dataset_io.routes`, `trajectory_height` | HTTP entry and request validation |
| `robot_registry`, `gripper_registry`, `height_registry`, `dataset_io.storage` | Shared registries and ownership; no compute-to-API imports |
| `full_job`, `gripper_job`, `progressive_job`, `scene_job` | Task orchestration, publication and immutable snapshots |
| `body_*`, `window_*`, `run_egoallo*`, `transfer_smplx` | Human prior and time-aligned window processing |
| `retarget_robot`, `gripper_retarget`, `refine_operation`, `support_reference` | Shared robot conversion and physical metadata |
| `scene_inference`, `segment_*`, `sam_*`, `export_scene_cloud` | Current-frame perception |
| `render_recording`, `gripper_render`, `mobile_render_*` | Robot preview and source-camera composites |
| `settings` | Deployment paths and subprocess Python paths |
| `audit_*`, `prebuild_smpl`, `smpl_library`, `prepare_nymeria_*` | Explicit maintenance commands, never automatic on import |

## Deployment configuration

The previous Linux layout remains the default for compatibility. Set
`ARENA_WORKBENCH_ROOT`, `ARENA_STORE`, `ARENA_CONFIG_DIR`, `ARENA_SOURCE_ROOT`,
`ARENA_EGOALLO_ROOT`, `ARENA_DA3_ROOT`, `ARENA_SAM_ROOT`, `ARENA_SAM_WEIGHTS`,
`ARENA_JAX_ROOT`, `ARENA_UMI_CALIBRATION` and the interpreter overrides
`ARENA_MOTION_PYTHON`, `ARENA_SCENE_PYTHON`, `ARENA_EGOALLO_PYTHON`,
`ARENA_DA3_PYTHON` as needed. The Linux GPU scheduler uses POSIX locks. Install
this package in each runtime or supply its `src` directory on `PYTHONPATH`.
Weights, vendor packages, licensed body templates and private recordings remain
external; configuring a path never downloads or starts anything.

No archived implementation is imported from `experiments`. Abandoned historical
background fusion, object mesh completion, WiLoR/video-finger recovery, old
contact estimators and `/umi-height` were removed. Their versions remain in Git.
The single production `body_frames` implementation and EgoAllo license were
preserved. This cleanup changes organization, not model weights or solver math.

## Verification

Run `python -m pytest` from the repository, then `npm run build` and `npm test`
inside `web`. All maintained Workbench checks live in `tests/workbench`; no
production test script executes during collection. Linux media tests require
`ffmpeg` and `ffprobe` on `PATH`. Two private integration checks are opt-in via
`ARENA_PRIVATE_INTEGRATION=1` on a configured compute host; they require the
registered source recording and robot assets. Pure unit checks need no weights.
A working test suite does not certify physical feasibility of robot motion.
