# ArenaLabs workbench: architecture, usage and deployment

This guide describes the current six-point web pipeline, as inspected on
2026-10-18. The repository also contains older standalone CLI pipelines and
research scripts. Their existence does not mean the website uses their old
endpoint policies. The production entry is
`experiments/manipulation4d/serve_workbench.py`.

## 1. What this system produces

An input recording becomes a synchronized robot manipulation episode with:

- generated robot joint positions and next-frame kinematic action references;
- a reconstructed human body prior, reusable across target embodiments;
- current-frame DA3 point clouds, SAM instance masks and visible-surface boxes;
- task/object semantic annotations proposed by Qwen;
- the original head/camera video and available wrist videos;
- a synthetic first-person view with the selected robot's own arms and hands;
- observed-finger-to-Wuji motion where observed finger landmarks exist;
- optional video contact estimates and geometric fingertip proximity;
- calibration, timestamps, provenance, raw intermediate arrays and checksums.

This is a kinematic conversion and visualization system. It is not a robot
controller, a dynamics simulator, measured force recovery or a guarantee of
object contact, grasp success, balance or safe execution. The separate Operator
page presents the Isaac-based demonstration; it is not the numerical engine
behind every manipulation conversion.

## 2. Inputs and source adapters

All online motion clocks are 30 Hz. Never infer metres versus centimetres,
quaternion order or coordinate axes from filenames.

| Source | Body observations | End effector | Notes |
|---|---|---|---|
| EgoDex | Recorded camera/head and bilateral hand poses | Wuji from observed 21 landmarks per hand | Human body completion does not replace the original hand targets |
| Nymeria | Calibrated camera/head and observed bilateral wrist references | No dexterous hand | Official recorded clocks, fisheye rectification and head/device alignment matter |
| UMI | Head/camera proxy plus measured bilateral EE trajectories | Fixed gripper housing reference plus opening | Ask for the first trajectory minimum height when no world height exists |
| Real robot | Camera/head plus measured bilateral EE trajectories | Gripper reference plus opening | Recorded source body, pelvis and legs are excluded from body inference |
| User package | Explicit normalized camera and bilateral observations | Declared Wuji, no hand, or gripper | Vendor files must first be converted to the upload contract |

A bare MP4 is insufficient for this calibrated operation-retargeting path.
Video-only motion/SLAM recovery is not silently substituted. See
[the upload specification](dataset-io.md) for all required array shapes.

Hands and grippers use different declared task frames: Wuji uses the palm
reference; the current gripper route uses the fixed housing-center reference
from the embodiment metadata. Finger opening is a separate variable. Changing
the reference requires the corresponding source-to-tool extrinsic; it is not
just renaming the displayed error metric.

## 3. Human reconstruction and six-point conditioning

1. Adapt source observations without changing the measured task trajectories.
2. Derive the EgoAllo head/CPF input from the declared camera relationship.
   The UMI lower-neck mount convention is an approximation, not an independently
   measured optical/head calibration. Saved camera poses remain the scene inputs.
3. Recover a body with EgoAllo. The released model uses SMPL-H internally,
   which is why the extended 16-beta SMPL-H model is required even though the
   resulting human asset is SMPL-X.
4. Transfer the recovered pose to SMPL-X using a constant shape fit. Save the
   transfer approximation and original SMPL-H result separately.
5. Supply AlphaMotion only pelvis, head, left/right hand and left/right foot.
   The skeleton tree/rest geometry remains model metadata; other source-pose
   channels are masked. Orientations and positions of the selected roles are
   encoded under the original normalization and coordinate conventions.
6. Decode/project to the selected robot and run the original robot refinement.
   The exact observed palm/gripper targets remain precision objectives.

AlphaMotion is a **single-frame** model. Frames are batched for throughput;
batching does not turn it into a temporal generator. Temporal continuity comes
from human windows, the saved body trajectory and the original refiner.

The common mask is in `src/greenwich_pipelines/input_policy.py`; the receipt is
`human-six-pelvis-head-hands-feet-v1`. The 2026-10-18 migration regenerated and
checked all 87 active source–robot caches: 58 manipulation/locomanipulation and
29 locomotion combinations. Old raw outputs remain history, not active results.

### Fixed windows and overlap

EgoAllo uses 300-frame windows with 60-frame overlap. All windows use a shared
world, floor and constant body shape. The first window also estimates shape;
that calibration is additional work, not a free operation. Thirty sampling
rounds, inner guidance and final post guidance remain enabled.

The last window is padded to the fixed shape; padding is excluded from valid
constraints and output. Window checkpoints retain hashes and dependencies.
Overlap observations constrain the next window; quaternion/contact stitching
is not the only continuity mechanism. Failed windows are retained and can be
resumed with matching provenance.

A second GPU can consume committed stable human frames while the next window
is being recovered. Frame ids and timestamps accompany those outputs. Final
conversion reuses a streamed decode only if model/target identity, integer
codes, frame id and timestamp match exactly. Final refinement still operates
on the assembled motion. This is not yet fully online window-by-window final
refinement or live physical control. With no spare GPU, the consumer does not
block the producer indefinitely; final conversion follows its normal path.

### Height and floor

The declared world floor is distinct from a robot display offset. Uploads with
known world height keep their original calibration/cache. UMI asks for a
minimum trajectory height before its first generation; that first calibration
becomes its default saved result. Subsequent custom height variants must not
replace the source default. Other sources expose the adjustment in the player.
Body prior identity must reflect changes to floor/height and observation policy.

## 4. Scene perception

The scene branch can run while body reconstruction runs on another GPU.

1. Decode source video according to its actual clock. Nymeria rectifies the
   recorded fisheye view before pinhole depth inference and uses calibrated
   head/device/camera transforms.
2. Run DA3 `depth-anything/DA3NESTED-GIANT-LARGE-1.1` at the current configured
   resolution. The normal visualization clock is 15 Hz, with source frame ids.
3. Qwen proposes visible manipulable objects and semantic descriptions. Proposal
   text and evidence are retained. It does not prove an object is correctly named.
4. SAM 3.1 produces temporal instance masks. Independent category groups can
   occupy separate GPUs; a category's temporal sequence is not arbitrarily split.
   Empty trackers restart only after an actual subsequent detection, retaining
   earlier output and changing instance identity rather than inventing continuity.
5. Lift masks through the same depth and camera transform to form instance-colored
   point clouds and 3D boxes. Exclude source hands, arms, controllers and grippers
   from scene geometry. Boxes enclose visible estimated surfaces.

The default display uses **only current-frame** points. It does not accumulate
old depth layers, reconstruct hidden object backs or display completed object
meshes. Camera calibration and depth scale are separate sources of error;
low robot hand-reference error does not establish scene alignment.

Measured-camera clips use calibrated batches with shared anchors. The fallback
without measured camera poses uses a joint uncalibrated prediction; independent
unregistered depth batches would introduce inconsistent scale. This fallback
can consume substantially more memory for long recordings.

## 5. FOV compositing and contact

Native arm membership comes from the robot tree; meshes and playback must bind
exactly. Dummy original hands are excluded when a replacement hand is attached.
EGL/OpenGL renders the robot through the source camera. SAM foreground masks
remove the source human/hardware and retain appropriate object occlusions.
DA3 supplies estimated scene depth for compositing.

The background fill is currently OpenCV **Telea CPU inpainting**, not a learned
video inpainting network. SAM is the mask provider. Identical source image/mask
pairs share a lossless fill cache across robot embodiments. Telea may struggle
with large holes; a plausible composite is not camera-calibration evidence.

Qwen video contact is an estimate sampled at its declared rate, with unknown
values retained for unobserved digits. Separately, a CPU KD-tree compares ten
observed fingertips with currently visible object points. This proximity score
is not a pressure measurement or calibrated contact probability. Missing source
fingers do not become valid pressure estimates. The gripper/no-hand branches
currently do not produce dexterous per-finger tactile data.

## 6. Publication, caching and scheduling

`full_job.py` and `gripper_job.py` coordinate independent branches. The display
publishes motion → scene → FOV → contact as prerequisites become available.
That publication order does not force all computation into one serial chain.

Caches have distinct scopes:

| Asset | Shared across | Identity/provenance |
|---|---|---|
| Human prior | Target robots | Observation/window/floor policy and saved hashes |
| DA3 / semantic / SAM scene | Target robots | Source frames, model/policy, worker hashes |
| SAM category | Matching frame/category requests | Pixel, prompt, model and worker identity |
| FOV background fill | Target robots | Exact image/mask bytes and fill implementation |
| Robot decode | Matching target/frame/code | Model descriptor, codes and time |
| Refined robot result | Same source/robot/reference policy | Source hashes, encoder/support policy |
| LeRobot ZIP | One immutable export | Saved modalities and per-file SHA256 |

The scheduler leases physical GPUs with cross-process file locks. The current
pool is 0–3 and admits one heavy stage per card, requiring roughly 40 GB free
and refusing cards with more than 1 GB already used by external work. This is
conservative; four GPUs can run four heavy stages, but small render stages do
not automatically share spare memory with a resident large model.

GPU execution time includes process startup, model loading, rendering and any
CPU work performed while holding the lease. Queue time must be reported
separately. See [the audit](performance-audit-20261018.md).

## 7. Use the website

1. Open Manipulation and choose a preset in **Data library**, or open **Upload data**.
2. For uploads, download the packer, provide the required observations/video and
   submit its ZIP. The normalized contract is checked before generation.
3. Select a robot and source-compatible end effector. A cached combination is
   marked even when only one robot exists for that recording.
4. Play motion while remaining modalities finish. Inspect palm/gripper errors,
   joint waveforms, instance boxes and the synthetic FOV.
5. Use **Download dataset** after generation completes. Packaging runs on CPU
   from saved artifacts and displays the ZIP link when ready.

The browser gets an HttpOnly/Secure/SameSite workspace cookie. Uploads and exports
are isolated by that workspace. Clearing cookies loses that browser's access.
There are no collaborative accounts, recovery/login workflow, per-user compute
quotas or a production abuse-control system in the current implementation.
Publicly reachable does not mean ready for unlimited anonymous GPU workloads.

## 8. HTTP interface

Use the same HTTPS origin as the application. Initialize a workspace cookie jar:

```bash
BASE=https://YOUR-WORKBENCH-HOST
curl --fail --cookie-jar workspace.cookies "$BASE/api/4d/library"
curl --fail --cookie workspace.cookies --cookie-jar workspace.cookies   -F file=@recording.zip "$BASE/api/4d/uploads"
```

Use the returned recording id and an id from `/api/4d/robots`:

```bash
curl --fail --cookie workspace.cookies -H 'Content-Type: application/json'   -d '{"scene_id":"UPLOAD_ID","robot_id":"h2","hand_id":"wuji"}'   "$BASE/api/4d/generate"
curl --fail --cookie workspace.cookies "$BASE/api/4d/jobs/JOB_ID"
curl --fail --cookie workspace.cookies -H 'Content-Type: application/json'   -d '{"scene_id":"UPLOAD_ID","robot_id":"h2"}' "$BASE/api/4d/exports"
curl --fail --cookie workspace.cookies "$BASE/api/4d/exports/EXPORT_ID"
curl --fail --cookie workspace.cookies -o episode.zip   "$BASE/api/4d/exports/EXPORT_ID/download"
```

Wait for `completed` at each dependent step. Upload limit is 2 GiB compressed;
validation also bounds expanded archives and NPZ members. The complete input and
output schemas, explicit units and LeRobot reading example are in
[dataset-io.md](dataset-io.md). Exported `action[t]` is the next frame's generated
joint reference, not the real robot's original action command.

## 9. Self-hosting: current deployment layout

Website users need **no SSH**. A system administrator or deployment service
needs access to install code, obtain licensed weights, configure the domain and
start workers. Do not distribute the owner's SSH private key to users.

The current Linux deployment uses this layout. Paths below are current defaults,
not a claim that every worker supports arbitrary relocation:

```text
/opt/arenalab/
  src/                  installed source package
  configs/              runtime.cloud.json and viewer.cloud.json
  private/              weights, model caches and private inputs
  store/                served generated assets (private assets are guarded)
  web/dist/             built frontend
/home/ubuntu/arenalab-experiments/
  egoallo/              upstream runtime, licensed body models and checkpoint
    venv/               EgoAllo Python / CUDA JAX environment
    runtime-jax061/     isolated CUDA plugins/cuDNN and persistent JAX cache
  da3/venv/             pinned DA3 environment
  manipulation4d/
    code/manipulation4d/ current workbench modules
    venv/               SAM/Qwen/rendering and dataset packaging runtime
    sam3/               separately installed upstream SAM source
    sam31-weights/      authorized SAM 3.1 weights
    body-jobs/          immutable source human assets
    scene-jobs/         immutable depth/mask workspaces
    scheduler/          leases, queues and timing events
    queue/              generation job records and retained logs
    user-data/          private uploaded originals and registry
    dataset-exports/    immutable complete episode packages
```

### Administrator preparation

1. Provision compatible NVIDIA hardware/drivers and Linux EGL. The audited host
   has four 48 GB RTX A6000 cards and 56 CPU cores. Current scheduler defaults
   are unsuitable for an 8–24 GB GPU without changing/testing admission rules.
2. Clone the code and install the core package as described in the README.
   Build the frontend with `npm ci && npm run build`; deploy its `dist/` content
   under the configured `/web/` asset route. Preserve runtime logos/navigation assets.
3. Prepare separate upstream environments for EgoAllo/JAX, DA3 and SAM/Qwen.
   Follow those projects' installation instructions and validate actual CUDA
   execution. The optional dependency groups in this repository are not a
   complete environment lock for every upstream project.
4. Obtain the weight/model files in [model-release.md](model-release.md).
   Configure `runtime.cloud.json` with the actual AlphaMotion directory,
   Hugging Face cache and CUDA device. The directory needs `config.json` and
   `model.safetensors`; runtime model loading defaults to offline.
5. Provide robot MJCF/URDF/meshes and the viewer roster. Valid limits, axes,
   anatomical roles, endpoint frames, palm/gripper transforms and mesh binding
   are required; arbitrary URDF upload cannot infer missing physical metadata.
6. Install the workbench modules and upstream resources at the documented layout,
   or explicitly refactor and test hard-coded paths before relocating them.
   Search with `rg '/opt/arenalab|/home/ubuntu' experiments/manipulation4d src`.
7. Run `serve_workbench.py` using the API environment, bound to loopback. Its
   current config path is `/opt/arenalab/configs/viewer.cloud.json`.
   Manage it with systemd or your own supervisor. Restarting interrupts active
   jobs: wait/drain the queue first. Completed stage checkpoints remain retained.
8. Reverse proxy HTTPS to port 8765. Forward cookies and `X-Forwarded-Proto`,
   set `/api/4d/uploads` to accept 2 GiB plus multipart overhead, stream request
   bodies and allow long uploads. Keep model/registry/raw-private paths denied.
   Do not serve private store directories through an unguarded static alias.
9. Validate one observed-data upload end-to-end, another robot from the same
   human prior, a complete export with the official LeRobot reader, and a
   cross-workspace request that must be denied.

`ARENA_WORKBENCH_ROOT`, `ARENA_STORE` in dataset storage,
`ARENA_DATASET_PYTHON` and `ARENA_SCHEDULER_ROOT` are useful overrides for isolated
tests. They are **not** a complete deployment configuration layer; some legacy
workers still use fixed production paths. Four-card API scheduling is currently
configured in code. A complete portable installer, container/lock files and
external persistent task queue remain release work.

For contributions, collaborators fork/clone the public GitHub repository and
submit changes normally. A trusted maintainer/CI deploys reviewed code to the
cloud. Uploading a recording to the website and deploying server code are two
separate permissions; neither requires giving every user shell access.

## 10. Troubleshooting and trust boundaries

- JAX says CPU: the GPU plugin is absent/mismatched or environment variables were
  set after importing JAX. Test in a new process; do not accept silent fallback.
- No GPU stage starts: inspect lease tickets and actual GPU memory. An external
  workload may intentionally make a card ineligible.
- Motion ready but FOV still running: perception/rendering can be slower; a
  healthy GPU model stage does not imply every CPU/video task is fast.
- Hand-reference error is small but the object is missed: inspect camera
  calibration, depth scale, object masks and fingertip geometry separately.
- Upload is rejected: verify 30 Hz, units, right-handed Z-up poses, rotations,
  video coverage, image dimensions and the declared end effector.
- A channel is missing: preserve its missing/unknown status. Do not fill it with
  another recording's output or zeros that look like measurements.
- A model/file changed: use a versioned work directory and retain raw priors.
  Do not put a new policy label on an old result.
- Public repository visibility does not supply a software/model license or the
  rights to redistribute upstream body templates, weights, robot meshes or data.
