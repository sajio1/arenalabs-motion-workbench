# Unified motion API 1.3

The website, Python SDK, CLI and `/api/v1` use the same workflow contract. Provided full-body motion enters the unified body retargeter directly. When no full-body motion is provided, synchronized head and two wrist observations feed native SMPL-H reconstruction through EgoAllo. Neither source completion nor solver tolerance is a public user override.

Both manipulation and locomotion dispatch to the unified GPU hybrid backend. The locomotion viewer invokes `python -m arenalab_workbench.unified_conversion --request request.json`; it no longer dispatches `greenwich_pipelines run`. GPU stages share cross-process scheduler leases with manipulation. This changes the selected algorithm, not the meaning of physics validation: playback remains a kinematic reference.

## Explicit size choice

`RunRequest.trajectory_scale` accepts `preserve`, `match_robot`, or omission:

- `preserve` retains original source metres and object dimensions.
- `match_robot` explicitly normalizes the whole interaction to the robot. Source body, operation trajectories and scene geometry share a uniform similarity. Robot geometry is unchanged. This is not the original object-size task.
- An omitted choice resolves to `preserve` only when preflight does not flag a known large mismatch. For a known large mismatch, `/plans` returns `executable=false`, `required_choice=trajectory_scale` and both options; `/runs` rejects with HTTP 409 and `detail.code=SCALE_CHOICE_REQUIRED`.
- Unknown morphology is reported as unknown. It never silently enables scaling. Explicit `match_robot` is rejected with `SCALE_ESTIMATE_UNAVAILABLE` until an estimate is available.

Preflight uses generic morphology metadata, not robot names. `morphology.scale_factor` is an estimated neutral-size ratio for the decision; the final applied scale comes from the numerical result. `source_policy`, `source_policy_version`, and `motion_policy_version` describe routing and policy. Plans are read-only previews; workers validate current inputs again.

An example body for both `/plans` and `/runs`:

```json
{"workflow":"manipulation","recording_id":"recording-id","robot_id":"robot-id","hand_id":"none","trajectory_scale":"preserve"}
```

Use a retained workspace cookie, and an `Idempotency-Key` on `/runs`. A changed scale mode is a changed request and requires a different key. Browser retries preserve the same key after network or server ambiguity; structured 4xx rejections release that rejected browser submission key. A session error does not silently submit again.

SDK and CLI examples:

```python
plan = client.plan(workflow="locomotion", recording_id="recording-id", robot_id="robot-id")
# Inspect plan["policy"]["morphology"] and choose explicitly if required.
job = client.run(idempotency_key="retained-key", workflow="locomotion",
                 recording_id="recording-id", robot_id="robot-id", trajectory_scale="match_robot")
```

```sh
arena-api run --workflow locomotion --recording recording-id --robot robot-id --trajectory-scale preserve --key retained-key
```

Export does not initiate a new size decision or reconstruction. Its optional `trajectory_scale` selects the saved generated variant. The manipulation viewer supplies the mode of the currently displayed robot result.

## Select generated outputs

After robot selection, the website presents a multi-select with **Select all available**. `/plans`, `/runs`, the SDK and CLI accept `output_modalities`, a nonempty unique list of `trajectory`, `point_cloud`, `contact`, and `inpaint_fov`. Omission retains the legacy behavior of generating all applicable outputs. A changed set is a changed request and needs a new idempotency key; ordering does not change request identity.

| Selected output | Required computation |
| --- | --- |
| `trajectory` | Supplied body or native SMPL-H recovery, followed by the existing unified refinement and robot export |
| `point_cloud` | Registered-camera perception; no robot refinement. Explicit `match_robot` also prepares the source body to estimate the common scene scale |
| `contact` | Observed hand stream and source-video contact estimation; no body recovery or robot refinement. Geometry proximity is optional when geometry is already available |
| `inpaint_fov` | Robot trajectory, scene geometry and the supported FOV compositor as internal dependencies |

Plans expose `outputs.requested`, `effective`, `dependencies`, and `available` (each entry has `id`, `available`, `reason`). Availability is derived from registered input files and calibration. A public simulation preview is not a source camera. Contact currently requires observed 21-landmark hands and camera input; gripper contact is unavailable. FOV currently supports the observed-hand and registered-gripper renderers. Motion-only HiPhi and GRAIL sources therefore offer trajectory generation only. Locomotion's motion viewer also offers trajectory only.

```json
{"workflow":"manipulation","recording_id":"recording-id","robot_id":"robot-id","trajectory_scale":"preserve","output_modalities":["trajectory","contact"]}
```

The equivalent CLI flag is `--outputs trajectory contact`. Python calls accept `output_modalities=["trajectory", "contact"]` with the same contract. All selections share reusable source, motion and perception artifacts; selected public scene manifests have separate identities.

Unavailable explicit choices produce a non-executable plan and HTTP 409 `OUTPUT_MODALITY_UNAVAILABLE` on `/runs`; empty, unknown and duplicate choices are invalid. Workers revalidate capability before starting. The public scene records `output_modalities`, `completed_outputs`, and `output_status`, and publishes only requested results. Intermediate dependencies remain reusable worker artifacts; generating FOV alone does not advertise trajectory or point cloud as selected outputs. Completion means that the requested set is ready, not that all four modalities ran.

Export's optional `output_modalities` selects the saved scene variant, just like `trajectory_scale`; it never starts reconstruction or filters an unrelated variant. Robot dataset export requires a selected, completed trajectory. Point-cloud/contact/FOV-only scenes remain viewable but are not robot-trajectory datasets.

## Result and display boundary

Jobs retain effective scale and source policy across progress polls even if an intermediate worker update omits them. Operation completion is separate from solver convergence, quality and physical validation. The API exposes solver status/convergence without treating a stopped solver as a success certificate.

The manipulation renderer accepts optional `source_transform={scale,anchor_xyz_m,world_up:"z",applied:false}`. It maps raw Z-up coordinates `[x,y,z]` to its existing display basis `[x,z,-y]`, then applies `a+s(p-a)` only to source cloud, human cloud, source object bounds and recorded-camera positions. Robot geometry and wrist reference/error traces are already in the numerical output frame and are not scaled again. Camera orientation and original RGB frames remain unchanged. `applied:true` prevents a second source transform. Locomotion's distinct viewer basis `[y,z,x]` is handled by its numerical exporter, not this manipulation helper.

Motion-only recordings can be selected without an RGB video. Their source is explicitly labelled “No source RGB”; no synthetic video is presented as an original camera stream.

## Validation scope

Independent API tests cover required/unknown morphology, both explicit choices in both workflows, idempotency separation, policy retention, exports, legacy helper compatibility and rejection of solver overrides. Viewer tests cover unified CLI dispatch, per-scale caches and four-card queue isolation with mocked subprocesses. Browser unit tests cover submission/session semantics and numeric source-transform parity. These tests do not execute a model or establish motion quality; deployed numerical smoke tests and visual checks are separate release evidence.

### Source display coordinates and camera projection

`scene.source_world_to_display` is an explicit proper 3×3 column-vector rotation from the numerical Z-up world into the scene Y-up display world. Ordinary manipulation uses `[x,z,-y]`; gripper export uses `[y,z,x]`. Missing metadata retains the legacy ordinary-manipulation convention. The frontend rejects reflections, scaling, or a mismatched up axis.

Raw `source_body` vertices and raw object poses use that rotation. Existing depth clouds, segmentation boxes and source cameras are already in display coordinates and receive only `source_transform`'s uniform similarity. The similarity anchor is converted by the same declared rotation. Robot meshes and reference traces already follow numerical world outputs and are outside the source transform group.

Both FOV compositors move source camera origins by the same similarity and multiply observed source depth by its scale. Camera rotations, intrinsic calibration, distortion mapping and original RGB remain unchanged; shared depth caches are never mutated. Clipping and source-depth occlusion tolerances use the scaled scene units. This preserves the original source projection; it is a normalized interaction, not a claim about the original object's physical dimensions.

CPU tests independently check pinhole projection and occlusion ordering for both bases at scales 0.58/1/1.4, including translated anchors and rotating cameras. Three.js tests check body/object placement and normals under the two rotations. These tests do not constitute GPU-rendered visual acceptance; isolated test-deployment QA is separate.
