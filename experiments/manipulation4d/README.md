# Manipulation workbench

The production entry is `serve_workbench.py`, which attaches `workbench_api.py`
to the shared Motion Lab server. `full_job.py` coordinates the stage scheduler.
There are no independent solvers for individual action labels or robot names.

## Runtime responsibilities

| Responsibility | Entry/module |
|---|---|
| User uploads, workspace ownership, LeRobot export | `dataset_io/` |
| Shared body prior | `body_prior.py`, `body_inference.py` |
| Resumable EgoAllo windows | `run_egoallo_windowed.py`, `window_plan.py` |
| Concurrent committed-frame conversion | `window_robot_stream.py`, `emit_body_window.py` |
| Six-point encoder and exact-code frame reuse | `src/greenwich_pipelines/input_policy.py`, `frame_cache.py` |
| Humanoid conversion and palm refinement | `retarget_robot.py`, `refine_operation.py` |
| Gripper source/body/scene/output | `gripper_source.py`, `gripper_body.py`, `gripper_scene.py`, `gripper_job.py` |
| Native wheeled topology | `wheeled_prior.py`, `wheeled_body.py`, `wheeled_job.py` |
| DA3/Qwen/SAM scene generation | `scene_job.py`, `scene_inference.py`, `segment_parallel.py`, `export_scene_cloud.py` |
| FOV compositing | `render_recording.py`, `gripper_render.py`, `mobile_render_job.py` |
| Video contact and geometric proximity | `contact_recording.py`, `proximity_recording.py` |
| GPU/process locks and stage publication | `gpu_scheduler.py`, `progressive_job.py`, `job_status.py` |

Other audit, calibration, surface-completion and experimental scripts are kept
as research history; they are not additional user generation pipelines.
The default viewer shows current-frame point clouds, without historical fusion,
completed object meshes or source-hand skeleton overlays.

The raw human prior, raw AlphaMotion output, refined robot motion and visual
estimates are distinct artifacts with hashes. Cache policy changes must never
be applied by changing a label on an old robot result. Use the explicit
`tools/rebuild_cached_outputs.py` maintenance command with an audited pair list.

Private runtime directories, inputs, models, generated caches and deployment
receipts stay outside Git. Defaults match the existing server; core worker
roots can be isolated using `ARENA_WORKBENCH_ROOT` for integration tests.
For input/output contracts and deployment prerequisites, see
[`docs/dataset-io.md`](../../docs/dataset-io.md).
