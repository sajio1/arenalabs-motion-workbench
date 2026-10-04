# URDF onboarding audit — 2026-10-04

Scope: actual local Motion Lab, locomotion and localmanipulation, A3 T3D0, Fourier GR3 dummy-hand model, and Tienkung 2 Lite. This is not validation of every variant of those brands. No production solver, weights or prior outputs were modified.

## Observed user flow

Original vendor URDF + all referenced meshes were packaged into ZIPs. All three original packages were rejected by the upload route with HTTP 400 because valid `../meshes/` paths are forbidden before their resolved destination is checked. Browser uploads reproduced this. Diagnostic packages changed asset locations only; kinematics, axes, joint limits, origins and masses were unchanged.

| Robot | Original ZIP | Portable resource layout | Motion conversion |
| --- | --- | --- | --- |
| A3 T3D0 | rejected | missing mesh during MuJoCo import despite successful resource lookup | rejected |
| GR3 dummy-hand | rejected | preview exported; metadata required | rejected |
| Tienkung 2 Lite | rejected | preview exported; metadata required | rejected |

Two SOMA actions (jog and crouch) and two different 15-second Nymeria excerpts were submitted for each robot: 12 requests, all HTTP 409 before solver execution. Browser selection of GR3 for an existing Nymeria slot also reproduced the refusal and retention of the previous result. No new robot motion was generated; endpoint accuracy, ground contact, self-collision and zero-shot motion quality remain untested.

Evidence is retained locally under `runs/urdf-onboarding-audit/`: original/portable packages, packages.json, user-flow-results.json and descriptor-probes.json. Screenshots are under validation/screenshots/urdf-onboarding-*.jpg. These private assets remain outside Git.

## P0: unblock universal onboarding

1. Resolve resource paths against the URDF location, then enforce that the final path stays inside the extracted package. Allow safe parent-directory references. Discover package:// roots in nested ZIP layouts rather than matching a few hardcoded locations.
2. Normalize MuJoCo compiler path behavior: imported URDF defaults strip mesh paths. A3 has no explicit compiler declaration; the uploader adds one but never sets strippath=false. The mesh lookup passes, then MuJoCo searches for pelvis_link.STL at the upload root. Keep meshdir and absolute resolved paths consistent. Report resource resolution and model compilation separately.
3. Replace the unconditional conversion_ready=false gate with real descriptor ingestion and persistence. Uploaded rows currently contain preview/report fields but no pipeline_row, model path or semantic descriptor that convert_task can use. There is no UI to complete the demanded metadata. Merely changing the boolean would therefore not fix conversion.
4. Normalize humanoid base topology. All three vendor URDFs lack a floating joint; both successfully imported uploaded models have zero free joints. Locomotion needs a moving base. Generate a floating-root model for a humanoid, preserve fixed-base and wheel topology appropriately, and use that same generated model for descriptor, refinement and playback. Do not apply a robot-name exception or a display-only root lift.
5. Share endpoint resolution across pipelines. Nymeria's fallback requires left/right substrings and wrist/ankle terms; it does not recognize end_effector, hand or l/r suffixes. Real GR3 and Tienkung MJCF descriptor probes failed with max() iterable argument is empty. Fullbody's resolver found end_effector/foot links for GR3 and hand/suffix ankle links for Tienkung. Resolve candidate chains using names, topology, joint axes and geometry, expose confidence and ambiguities, and persist the four physical endpoints once.

## P1: establish reliable frames and feedback

6. Reconcile URDF joint axes with endpoint semantic/TCP frames. A URDF frame is not automatically a palm/foot frame. Use explicit tool frames or reliably infer neutral frames from topology/geometry, then show an axes preview when ambiguous. Nymeria currently requires palm_frames and its runner rejects missing ones. Declaring every robot to share the same axes would recreate the previous wrist-orientation problem. This is fixed physical metadata, not per-motion tuning.
7. Make onboarding statuses actionable: resource failure, invalid model, ambiguous endpoint, ambiguous frame, ready. Current A3 failed compilation is still announced as added; the robot tray labels failures generically as Metadata required. Explain the concrete missing item and provide the completion UI.
8. Separate kinematic readiness from visualization. Missing external STL files should not prevent descriptor construction when inertial/collision primitives provide a usable kinematic model. Show missing geometry explicitly. A URDF cannot carry external mesh bytes by itself; accepting a directory/ZIP is necessary for an accurate visual model.
9. Validate the generated descriptor with independent FK before model inference: rest pose, axes, joint ordering, moving root, endpoint selection, units and joint limits. Then run the same selected inputs on all three robots and recompute wrist pose/ground/temporal checks from the actual final exported trajectories. Use the user's 1 cm and 15 degree wrist tolerances. No stricter mean/p95 thresholds or default quality gates are introduced by this audit.

## Additional source-level findings, not motion-validated

- Fullbody and Nymeria have separate name resolvers and explicit-frame requirements, so the same upload has inconsistent compatibility across pages.
- visible_geoms currently retains both visual and collision meshes when both exist on a body; the uploaded GR3 has 59 geoms. Surface and collision representations need separate purposes, and identical visual duplicates should be deduplicated for rendering.
- A3 vendor MJCF independently references a missing external terrain include. A scene dependency should not be required to ingest a robot-only URDF. This failure belongs to that MJCF asset path, not evidence that the A3 URDF kinematics are unusable.
- No force/torque feasibility or simulation evaluation was conducted. Fixing import errors does not certify every movement as physically executable.

## Acceptance for the next implementation

For these three robots, an unedited valid vendor package should upload, resolve assets, create the descriptor and required floating root, and run both selected SOMA actions and both Nymeria excerpts without a robot-specific solver file or policy weights. Test fixed root/renamed links/nested packages and ambiguous TCP cases explicitly. A recognizable open-chain humanoid should be automatic; an ambiguous or mechanically underspecified URDF should receive a small, concrete frame/topology confirmation instead of a generic metadata dead end. Arbitrary URDFs do not mathematically guarantee sufficient task semantics or achievable poses.

## Implemented repair and validation — 2026-10-04

The blocking upload path is repaired for locomotion/localmanipulation. Safe parent-directory and nested package:// references are resolved within the extracted package; MuJoCo retains resolved mesh paths. A successful humanoid upload creates model.xml, descriptor.json and descriptor.npz, adds a geometry-derived floating base when omitted, resolves four endpoint links, infers fixed neutral wrist axes, and persists its pipeline row. The same upload appears in both humanoid pages. Names/IDs do not select solver weights or policies. Existing H2/G1 mechanical metadata and the SOMA 100+100 / Nymeria solver distinction are preserved.

Wrist axes are inferred physical estimates, not certified TCP calibration. The inspection dialog exposes endpoint/frame metadata for correction. The API checks distinct links and finite nondegenerate axes, rejects edits while conversions are active, invalidates affected caches and persists revisions. Successfully ingested robots now have a ready state; resource/model failures have specific messages. Completed playback caches survive service restart and are keyed by source fingerprint, source window and physical descriptor/model identity.

GR3 revealed another name dependency in the Nymeria refiner: a shoulder had to contain the literal word shoulder. Existing named shoulder behavior is preserved; otherwise the arm branch is found from the two wrist ancestry chains' common trunk. No GR3-specific branch was introduced.

The earlier Tienkung Lite URDF contains only arm chains through the elbow and no wrist links. It remains unsupported for four-wrist/foot input rather than silently using elbows as wrists. The successful Tienkung test uses the locally available complete tiangong2dex_29.urdf (39 links, 38 joints including fixed links), with explicitly defined TCPs. These are different robot variants.

### Actual conversions

A3 T3D0, GR3 dummy-hand and complete Tienkung were uploaded from vendor ZIPs without editing their mesh paths. Each ran two SOMA clips (147-frame jog, 150-frame crouch) and two Nymeria clips (240-frame historical pan/preparation excerpts): 12 completed conversions at 30 Hz. The 500 current 15-second Nymeria library windows were not all converted by this test. Failed earlier GR3 attempts and immutable model priors remain preserved. All 12 playback caches were verified after service restart. Forty-nine Python tests and the TypeScript/Vite build pass.

Independent MuJoCo endpoint probes were compared to the saved descriptor FK. Maximum discrepancy was below 0.001 mm for A3/Tienkung and below 0.574 mm for GR3. This validates motor/mesh binding, not target-tracking accuracy. Visual browser verification loaded the three uploaded robots together on both pages.

| Nymeria clip | Robot | Max wrist position error, L/R mm | Max wrist orientation error, L/R degrees | 10 mm / 15 degrees |
| --- | --- | --- | --- | --- |
| pan | A3 | 3.62 / 5.35 | 30.11 / 19.24 | fail orientation |
| preparation | A3 | 50.06 / 46.90 | 172.53 / 172.98 | fail both |
| pan | GR3 | 3.31 / 3.67 | 34.46 / 30.28 | fail orientation |
| preparation | GR3 | 5.44 / 9.85 | 47.96 / 154.23 | fail orientation |
| pan | Tienkung complete | 3.66 / 4.42 | 44.31 / 33.01 | fail orientation |
| preparation | Tienkung complete | 5.44 / 3.02 | 167.64 / 17.46 | fail orientation |

These angular errors are also present in the saved physical FK, not just the renderer. The upload rejection is fixed, but strict wrist-pose quality is not achieved on these samples. No new automatic acceptance gates, per-robot weight changes or simulation claims were added. The next quality task is to distinguish inferred-frame calibration error, unreachable endpoint combinations and the existing soft optimizer's orientation/continuity tradeoffs; this audit does not claim those causes are resolved.

Detailed private results are in runs/urdf-onboarding-audit/final-validation.json; screenshots are validation/screenshots/urdf-fixed-locomotion.jpg and urdf-fixed-nymeria.jpg.
