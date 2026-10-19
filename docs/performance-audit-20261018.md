# Pipeline performance audit — 2026-10-18

Subsequent execution measurements: [AlphaMotion cache fix](alphamotion-speed-20261018.md)
and [SAM feature/session reuse](sam-speed-20261018.md). The observations below
describe the earlier audit baseline.

## Scope and evidence

Inspected the deployed workbench workers and repository sources, the GPU lease
history from the six-point migration, and an isolated before/after FOV render.
The host has 4 × RTX A6000 (48 GB) and 56 CPU cores. This is an audit of the
current web path, not a claim that every retained research script is optimized.

Evidence committed with this report:

- [`performance-20261018.json`](../validation/performance-20261018.json): exact-output FOV benchmark, depth-access timings, fresh CUDA-JAX probe and working NVENC probe.
- [`migration-stage-timings-20261018.json`](../validation/migration-stage-timings-20261018.json): completed lease events whose job ids belong to the final 58-job migration receipt.
- [`audit_pipeline_performance.py`](../tools/audit_pipeline_performance.py): repeatable, read-only timing summarizer; no inference or cache modification.

The timing sample mixes cache reuse, source lengths and target robots. A stage
lease includes subprocess startup/model loading and CPU work under that lease;
it is **not pure GPU kernel time**. Cache hits without leases are absent. Stages
overlap, so adding stage medians does not give end-to-end latency. Original
failed attempts remain separate and are excluded from completed-event statistics.

## Applied fixes

### 1. Repeated whole-sequence decompression in gripper FOV rendering

`gripper_render.py` retained `np.load(depth.npz)` as an `NpzFile` and indexed
`archive['depth'][frame]` inside the video loop. An NPZ member is decompressed
on every member access; it is not lazy frame-level random access. The renderer
therefore inflated an entire sequence for each rendered frame. This was
avoidable CPU work performed while an entire GPU remained leased.

The fix loads only `depth`, `intrinsics` and `ids` once into arrays, closes the
archive, and uses ordinary array indexing. It also closes/materializes the
camera archive once. No depth values, model, masks, cameras, trajectories,
render settings, fill algorithm or codec were changed.

On the 539-frame real-robot bottle-lid sequence (303,425,644-byte depth archive),
the first 30 frames were rendered in isolated output folders with the same
native robot/gripper geometry and existing source-background cache:

| Measurement | Before | After |
|---|---:|---:|
| Six depth-member accesses / one materialization | 5.263 s | 0.877 s |
| 30-frame complete FOV subprocess and output | 40.243 s | 15.204 s |
| Decoded RGB SHA256 | `9ffc9eceed9c284523496ed2cf382478add20f2475aece60688a69c799a71fe4` | Same |

The FOV sample is **2.65× faster**, with bitwise-identical decoded video frames.
This is one sequential before/after test on one host, not a confidence interval
or a whole-pipeline benchmark. The new renderer retains the needed depth arrays
in CPU memory for its lifetime. It trades that bounded per-recording residency
for eliminating repeated decompression/allocation. It does not hold model weights
in RAM beyond their existing lifetime.

Affected routes: UMI/real-robot gripper composites and routes that delegate to
`gripper_render.run`, including compatible mobile composites. EgoDex humanoid
`render_recording.py` already materialized its depth archive once, so this change
does not accelerate that route by the same factor. Existing cached videos are
not needlessly regenerated for an execution-only fix.

### 2. Honor an explicitly selected CUDA device in JAX setup

`egoallo_runtime.gpu_environment(base)` previously read `CUDA_VISIBLE_DEVICES`
from the parent process even when the caller provided an explicit environment.
That could overwrite a scheduler-selected device. It now respects the supplied
base environment; default process behavior is unchanged. The regression test
sets parent device 3 and caller device 1 and verifies the caller's lease survives.

An EGL context reported `NVIDIA Corporation` and `NVIDIA RTX A6000/PCIe/SSE2`,
confirming hardware rendering rather than a software rasterizer.

A fresh upstream EgoAllo process reported `jax.default_backend() == 'gpu'` and
`CudaDevice(id=0)` under its visible-device mapping. The current runner already
rejects CPU JAX; no silent CPU fallback was found. The matching CUDA JAX plugin,
cuDNN library path and persistent compilation cache are already configured.

## Actual device placement

| Stage | Current compute | Assessment |
|---|---|---|
| Video decode, frame selection, lens remap | CPU/OpenCV | Ordinary preprocessing; repeated seek is already avoided for short forward steps |
| EgoAllo denoising | CUDA PyTorch | 30 rounds retained |
| EgoAllo constraint guidance | CUDA JAX | Fresh-process probe passed; fixed shape and compilation cache retained |
| SMPL-H/SMPL-X posed-body evaluation | CUDA PyTorch | Small shape fitting / quaternion-format conversion still on CPU |
| AlphaMotion encode/decode/project | CUDA PyTorch | Six-point visibility; single-frame model batched across frames |
| Robot operation-refinement objective / FK / gradients | CUDA PyTorch + CUDA Graph | CPU SciPy L-BFGS-B still controls optimization |
| DA3 | CUDA PyTorch | Calibrated batches use shared anchors; repeated anchor evaluation costs remain |
| Qwen discovery / hardware classification / contact | CUDA PyTorch | Separate processes may reload the same model |
| SAM 3.1 mask inference/propagation | CUDA PyTorch | Category groups parallel; tracker recovery does not invent observations |
| Cloud lift, labeling, packing | CPU NumPy | Mostly vectorized geometry and serialization, not an unaccelerated neural model |
| Native arm/hand image render | GPU EGL/OpenGL | Python per-node pose updates and CPU readback remain |
| Removed-human/hardware fill | **CPU OpenCV Telea** | Significant potential bottleneck for large masks; cached per exact image/mask |
| Video encoding | **CPU libx264** | System FFmpeg successfully encoded with `h264_nvenc` in a real probe; default not changed |
| Fingertip proximity | CPU SciPy cKDTree | Ten queries/frame; building GPU point structures is not automatically faster |
| LeRobot Parquet/hash/ZIP / video packaging | CPU, limited thread pool | No neural inference; ZIP stores already compressed streams without recompressing |

A CPU controller is not evidence that GPU gradients are unused. Conversely,
CUDA Graph capture does not make the entire SciPy optimization GPU-resident.
Each objective callback still exchanges optimization state/loss/gradient with
the CPU. Replacing L-BFGS-B with Adam or a different GPU solver changes the
solution path and requires motion-quality validation; it is not a safe toggle.

## Selected migration observations

These are completed **execution leases**, with queue delay shown separately.
Samples come only from final selected migration job ids, not all old experiments.
Cache reuse explains why counts are lower than the number of converted outputs.

| Stage label | n | Execution median | Queue median | Execution maximum |
|---|---:|---:|---:|---:|
| `retarget_robot.py` | 16 | 57.7 s | 84.7 s | 129.1 s |
| `gripper_retarget.py` | 9 | 48.0 s | 258.1 s | 64.6 s |
| `render_recording.py` | 13 | 28.1 s | 150.9 s | 241.8 s |
| `gripper_render.py` (before fix) | 10 | 149.3 s | 339.0 s | 583.2 s |
| SAM category worker | 11 | 139.9 s | 247.2 s | 585.3 s |
| `contact_recording.py` | 1 | 570.6 s | 268.0 s | 570.6 s |
| `scene_inference.py` (mixed depth/discovery labels) | 3 | 20.5 s | 393.7 s | 148.6 s |

A 570-second contact sample is a real outlier, not a typical-rate claim. Generic
`scene_inference.py` lease labels currently merge depth and discovery; use the
per-scene `depth.json` and stage receipts to separate them. More precise uniform
stage names and timing spans are useful observability work still to do.

## Remaining opportunities, in priority order

1. **CPU fill while reserving a whole GPU.** FOV owns a GPU lease through render,
   CPU Telea, readback and x264. Prepare source-only fill work in a bounded CPU
   pool before/alongside GPU rendering, with per-frame backpressure and the same
   exact cache identity. Then measure both latency and GPU slot occupancy. Do
   not spawn unbounded processes or copy every video frame into RAM.
2. **Hardware video encoding.** NVENC is actually available on this host. Add a
   shared encoder option with capability checks and an x264 fallback; compare
   decoded quality/PSNR, frame counts, timestamps, seekability and LeRobot decoder
   behavior. NVENC and x264 rate-control settings are not interchangeable, so
   changing the default was not bundled into the bitwise-preserving fix.
3. **Warm model workers.** Qwen and SAM can repeatedly load in separate processes.
   Long-lived workers per isolated runtime could remove cold starts. They need
   memory-aware scheduling, cache/revision ownership and failure recovery. The
   current scheduler excludes already-occupied GPUs; simply keeping models
   resident would make those cards ineligible without coordinated changes.
4. **Conservative exclusive leases.** All heavy jobs normally request 40 GB free
   and refuse external occupancy above 1 GB. A small EGL renderer blocks a whole
   card. Sharing by measured resource classes could improve throughput, but must
   respect actual EGL/CUDA device mapping, memory peaks and other users' jobs.
5. **CPU/GPU optimizer synchronization.** Capture reduces Python/autograd launch
   overhead but not SciPy's CPU line search. Benchmark same-objective GPU solver
   candidates against end-effector error, feet, continuity and convergence before
   choosing a replacement. Keep the original model prior immutable.
6. **Long-recording visual memory.** The fixed 300-frame human windows do not make
   every visual stage constant-memory. DA3 preparation currently loads the whole
   sampled video list and retains predictions; uncalibrated DA3 can be global.
   Stream calibrated inputs/output storage before advertising unbounded videos.
7. **Avoid unnecessary cache reads/copies.** Weight/source hashing and final
   exports are CPU/I/O work. Use immutable identities carefully; do not replace
   content validation with filename-only hits. Per-frame cached decoder assembly
   still makes many small transfers and is a potential micro-optimization.

Already implemented before this audit: CUDA JAX, persistent JAX compilation
cache, fixed body windows, committed-frame AlphaMotion overlap, exact-code decode
reuse, CUDA Graph objective execution, source-body sharing across robots,
parallel body/scene branches, category-level SAM parallelism and FOV background
reuse. None of these imply all stages are saturated or all model processes warm.

## Reproduce timing summaries

```bash
python tools/audit_pipeline_performance.py   --root /path/to/manipulation4d   --jobs selected-job-ids.json   --output audit.json
```

`selected-job-ids.json` is an array of job-id strings. Omit it to inspect all
retained scheduler history; that broader report is not the table above. The
audit reads metadata only. It does not upload recordings, run models, terminate
other work, change solver iterations or rewrite existing caches.

The isolated render comparison kept the active 539-frame result unchanged and
rendered 30-frame diagnostic copies. Raw model outputs and source trajectories
were not modified. Execution-only fixes are installed for subsequent worker
processes; a new full cache migration is unnecessary when rendered pixels and
model policies are unchanged.

Reference for the fill implementation:
[OpenCV inpainting API](https://docs.opencv.org/4.x/d7/d8b/group__photo__inpaint.html).
