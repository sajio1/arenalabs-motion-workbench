# ArenaLabs Platform API v1 — website integration 1.1.2

Base URL: `https://arenalab.192-222-53-217.nip.io/api/v1`

- Interactive OpenAPI: `/api/v1/docs`
- Machine-readable contract: `/api/v1/openapi.json`
- Current supported behavior: `/api/v1/capabilities`

This release packages the **existing platform**. It does not change retargeting,
native SMPL-H reconstruction, shared manipulation refinement, foot policy, or
Locomotion's existing solver. The recalibration proposal is deferred.

The current website now consumes this API on Manipulation, Locomotion and
Operator. Both motion workbenches use `web/src/platform-api.ts` for workspace,
submission identity, pagination, status and errors. Operator's maintained source
is in `web/operator/`; each frontend build includes it under `/web/operator/`.
Business/data requests use `/api/v1`; CSS, JavaScript, logos and renderer modules
remain static application assets. Existing bookmarks with `/assets/.../scene.json`
are translated to versioned media delivery without changing the stored manifests.

## GitHub scope and merge order

This API change is stacked on the native SMPL-H change in
[PR #1](https://github.com/sajio1/arenalabs-motion-workbench/pull/1).
The `codex/platform-api-v1` branch contains that commit as an ancestor. Its PR
initially targets `codex/native-smplh-only` so reviewers see only the API diff.
Merge PR #1 first, then change the API PR base to `main` before merging it.

This change publishes the control API, Python SDK/CLI, direct application-service
bindings, and API transport for all three website pages. Numerical refinement,
GRAIL source adapters, calibration and articulated-export format changes are
separate work. Installed backend versions determine those behaviors. A coworker
can branch from `codex/platform-api-v1` to include both changes; conflicts with
other work must be checked against its actual published branch.

The customer run contract accepts workflow, recording, robot and explicit
end-effector selection. It does not expose arbitrary solver weights, execution
stages or server paths. Internal service bindings are implementation details.

## Scope

| Capability | API |
|---|---|
| Initialize an SDK/browser workspace | `POST /sessions` |
| Paginated recordings | `GET /recordings?workflow=manipulation&limit=50&offset=0&q=...` |
| Current Locomotion/legacy catalogs | `GET /catalogs/{page}` |
| Recording upload template and ZIP import | `GET /recordings/template`, `POST /recordings/import` |
| Robot roster and URDF/ZIP ingestion | `GET /robots`, `POST /robots/import/{page}` |
| Existing physical metadata and robot deletion | `PATCH /robots/{robot_id}/metadata`, `DELETE /robots/{robot_id}` |
| End-effector roster and package ingestion | `GET/POST /end-effectors` |
| Existing trajectory-height feature | `GET /recordings/{scene_id}/height-preview`, `POST /height-adjustments` |
| Source video and RGB-D packet | `GET /recordings/{clip_id}/source-video`, `GET /geometry/frame` |
| Preflight | `POST /plans` |
| Manipulation, Locomotion, dataset export | `POST /runs` |
| Batch submissions | `POST /batches` |
| Unified status, retries and events | `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/retries`, `GET /jobs/{id}/events` |
| Published playback and quality data in manifests | `GET /jobs/{id}/playback` |
| Content-identified result references | `GET /jobs/{id}/artifacts`, `GET /artifacts/{id}` |
| Range/ETag artifact downloads | `GET /artifacts/{id}/content` |
| Viewer media, models and manifests | `GET/HEAD /media/{relative_path}` |
| Export ZIP download | `GET /jobs/{id}/download` |
| Operator recorded episodes | `GET /simulation-episodes`, `GET /simulation-episodes/{id}` |
| Operator playback and asset reads | `GET /simulation-episodes/{id}/playback`, `GET /simulation-episodes/{id}/assets/{kind}` |
| Operator shared robot geometry/metadata | `GET /operator/resources/{kind}` |

All old routes remain available. Import and metadata aliases keep their current
request schemas (some use `scene_id`); consult OpenAPI instead of guessing.
Robot import retains the existing page-dependent ingestion rules. Imported
hands may be `needs_mapping`; saving a hand is not proof that it can be driven.

Operator collections preserve the existing `review` modes using `collection`:
`default`, `single-v29`, `kitchen-v29b`, `scene-v30`, `source-demo`, `1`.
Each episode contains original camera URLs/offsets, result and report URLs,
and available Blender/RTX download URLs. Missing camera entries remain absent.
Binary gzip motion stays gzip bytes (no HTTP Content-Encoding that would cause
double decompression). Episode asset URLs include a revision; a changed source
returns 409. Original status labels and task evidence are preserved.

## Workspace

Create a session and retain its `arena_workspace` cookie. The SDK manages the
cookie for its lifetime. For future processes, retain that cookie securely and
pass it as `workspace_token`. Do not publish it in examples or output logs.

Jobs, artifacts and exports belong to the submitting workspace. An unrelated
workspace gets 404. The current robot and end-effector registries remain shared;
this release does not introduce project accounts or finer-grained roles.

```python
from arena_api.client import ArenaClient

with ArenaClient("https://arenalab.192-222-53-217.nip.io") as api:
    api.session()
    print(api.recordings(limit=10))
    print(api.robots())
    print(api.end_effectors())
```

## Run current Manipulation

```python
spec = {
    "workflow": "manipulation",
    "recording_id": "grail_pickup_ground__alcohol_0__001",
    "robot_id": "custom_95e90092ed",  # X2 in the current roster
    "hand_id": "none",              # explicit selection is required
}

plan = api.plan(**spec)
job = api.run(idempotency_key="my-grail-x2-001", **spec)
job = api.wait(job["id"], timeout=3600)
if job["status"] == "succeeded":
    playback = api.playback(job["id"])
    artifacts = api.artifacts(job["id"])
else:
    print(job["status"], job["error"])
```

`plan` is a metadata preflight, not a frozen execution revision. The existing
worker adapts the source and revalidates detailed inputs; native SMPL-H is used
where body reconstruction is required. Numerical
inputs, solver weights, Python module paths, command lines and arbitrary local
paths cannot be supplied through `/runs`.

For Locomotion use `workflow="locomotion"` and omit `hand_id`. Source IDs and
eligible robots come from `/catalogs/locomotion`.

For a dataset ZIP use `workflow="export"`, the generated Manipulation recording
and robot, and omit `hand_id`. Export delegates to the installed dataset
exporter and does not run inference. Available files and supported source types
depend on that exporter; this API change does not add a new export format.

```python
export = api.run(
    idempotency_key="my-grail-x2-export-001",
    workflow="export",
    recording_id=spec["recording_id"],
    robot_id=spec["robot_id"],
)
export = api.wait(export["id"])
if export["status"] == "succeeded":
    api.download(export["id"], "grail-x2.zip")
```

Use this inside the same client context as the session example. Existing target
files are not overwritten. Download uses a temporary file and publishes it only
after the complete response arrives.

## Jobs and errors

`POST /runs` returns 202 and the same `JobView` shape for queued work and cache
hits. A cache hit still gets a durable API job ID. `Idempotency-Key` is required.
Reusing a key with different inputs returns 409; reuse the same key after a
network timeout. A deliberate retry uses `/jobs/{id}/retries` with a new key.

`POST /batches` accepts up to 100 items. Its key binds the complete ordered batch.
Per-item identities permit resuming partial submission without duplicates.

States: `dispatching`, `queued`, `running`, `succeeded`, `failed`, `interrupted`,
`outcome_unknown`. Failed submission remains in the workspace's job registry.
If dispatch outcome cannot be determined, the API does not blindly resubmit.

The SQLite WAL registry persists identities and observed events. Current backend
schedulers still own execution. A backend restart can interrupt work; a durable
API record is **not** a guarantee that a GPU process resumes automatically.
List responses show last observed state; polling an individual job refreshes it.

The API binds directly to the running application service functions. It does not
make internal HTTP requests to the old endpoints. The same queue objects and
worker functions serve both API versions, so there is no second scheduler or
second solver policy. Import/resource routes also share their original typed
validation. Generation preserves `revision`, readiness flags and phase fields
so the browser still displays partial motion while FOV/geometry finishes.

Source end-effector requirements come from the same queue function used during
generation. A dataset label is insufficient: measured real grippers and GRAIL
appear under Real robot but require different end effectors. Height variants
inherit their source's requirement and workspace authorization.

The browser retains a submission key through ambiguous network errors and page
reloads. It displays interrupted/unknown outcomes as failures, not success, and
does not silently create a replacement task. Export downloads stay under the
versioned API; they no longer redirect the browser to `/api/4d`.

SSE uses `Last-Event-ID` or `after` for reconnects. Events are observed status
changes, not an invented complete history of every numerical iteration.

Safe cancellation of shared legacy stages is not implemented. The cancel route
returns 409 and does not kill processes. Independent arbitrary-stage execution,
new calibration, and live physics rollout report unsupported capabilities.

Standard request errors use `application/problem+json`; validation failures are
422, missing workspace 401, inaccessible job 404, and conflicting input 409.
Unexpected dispatch errors are logged server-side and surfaced as unknown
outcomes. Private server paths and credentials are not accepted as API inputs.

## Results and physics

`succeeded` means the existing operation finished, not that its optimizer
converged or that the motion is dynamically valid. Playback includes the existing
robot solver and quality fields. Physics status remains `not_run` for generated
motion. Operator responses retain recorded evidence without rerunning physics.

Playback manifests currently reference progressive legacy assets. They are not
fully immutable run snapshots. Artifact indexing records a hash and ownership;
if the underlying legacy file changes, downloading the old reference returns
409 rather than silently returning different content. Large files support
Range; ETag enables conditional download.

`GET /jobs/{id}/playback` also returns `selected_robot_id` and `assessment`.
The assessment separates operation completion, convergence, final tracked-point
quality, source routing, reconstructed body model, stance/contact diagnostics,
and physical validation. Missing solver or support reports stay null. Contact
diagnostics identify their refiner-report source; they are not measurements.
No automatic quality acceptance gate is introduced.

Exports use the existing dataset exporter. The API supplies workspace ownership,
job identity and download delivery; it does not change joint order, actuator
semantics, robot descriptors or the exported coordinate convention. Inspect the
backend's manifest and quality reports before consuming a motion. Kinematic
playback is not a validated physics controller.

New exports use `arenalabs-lerobot-v3.0-4-native-reference` for every input
source. EgoDex, HiPhi, GRAIL, Nymeria and recorded gripper inputs do not select
different downstream motion schemas. `extensions/arenalabs/native_reference/`
contains a versioned manifest, exact 30 Hz relative clock, original source
timestamps, named body and attachment coordinates, floating-base pose, derived
velocities and named world link poses. Native references use right-handed Z-up,
metres, wxyz quaternions, radians for hinges and metres for slides. Body and
attachment arrays always exist, including empty attachment arrays. Body-model
source family remains provenance; clients do not decode SMPL to play a robot.

`extensions/arenalabs/manifest.json.modalities` always contains `trajectory`,
`point_cloud`, `contact` and `inpaint_fov`, with `ready` or `not_selected` status
and package-relative paths. Missing requested outputs fail export. The exported
`validate_dataset.py` checks all hashes, clocks, full named joint coverage,
LeRobot state/action parity, native validation receipts and modality references
without cloud access or a robot model. It runs before finalization; ZIP CRC and
file inventory are checked before the `.partial` archive is atomically renamed.
Only a completed archive returns `ready: true`. Old archives remain immutable;
new exports with missing native provenance fail instead of silently returning a
visual-only package. The same receipt is exposed by `/api/v1/jobs/{id}` as
`result.ready` and `result.readiness`.

The included MuJoCo and Isaac adapters require the customer's matching local
MJCF or URDF/USD. They map joints by name, validate topology/FK and reject
missing, duplicated or incompatible coordinates. Replay defaults to body-only
and succeeds without hand models. Attachment
channels remain in the same package as optional data. Explicit `--with-hands`
requires the matching local assembly; its absence then produces a clear error.
Reports distinguish body replay from full-assembly validation. Physical robot
descriptions are not added to the cloud or download
by these adapters. Existing visual GLB/USD assets remain part of the visual
replay, so this is not a claim that an existing visual export hides geometry.

`ready` covers complete kinematic reference data. It does not certify actuator
commands, contact stability or controller tracking, and does not change
`physics_status: not_run`. See [native reference usage](native-reference.md).

## Deployment

`python -m arena_api.server` attaches `/api/v1` to the existing workbench app.
`ARENA_API_DATABASE` overrides its SQLite path. `ARENA_OPERATOR_ROOT` locates the
existing Operator catalog; missing resources return 404/503 rather than fake data.
Existing GPU allocation and all algorithm source precedence are retained.

Only one API control process is used with the current legacy registries. Do not
scale this deployment with multiple uvicorn workers until the underlying
schedulers/registries are replaced with process-safe equivalents.

This release covers the current website functions. It does not introduce a new
solver, an independent arbitrary-stage DAG, safe cancellation or live Operator
physics execution. Those were not current website capabilities.

Install the API extras with `pip install -e ".[api]"`. Private model weights,
recordings and Operator episode assets stay outside Git. Operator's viewer also
requires the existing `/isaac/cache` renderer modules and the site's shared
navigation assets. Keep the site's public `/operator` route mapped to the built
Operator page. This change does not replace deployment routing or algorithm
module-path precedence.

The live service already reports API version 1.1.2. Publishing this PR does not
redeploy it or replace the active numerical workers.

## Validation of this branch

- Backend/API and workbench suite: **56 passed, 8 skipped**. Skips cover optional
  Zarr, POSIX-only workers on Windows, and unavailable private integration data.
- Browser logic suite: **25 passed**, including session sharing, idempotent retry,
  URL normalization and interrupted/unknown job states.
- TypeScript compilation and production frontend build: **passed**, including
  copying the maintained Operator viewer into the distribution.

These are local contract/integration and browser-logic checks. This branch was
not subjected to a fresh GPU conversion, actual browser end-to-end run, or
physics rollout. Successful API jobs do not conceal motion-quality failures;
solver status and missing modalities remain part of the existing result.
