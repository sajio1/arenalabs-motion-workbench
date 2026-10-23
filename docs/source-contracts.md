# Recording source contracts

The raw library displays the declared input representation. Dataset names and
action titles must not determine whether an input is a robot, SMPL parameters,
a recorded skeleton, or camera/wrist observations.

Public source metadata includes `source_format`, `source_family`,
`source_body_family`, `source_evidence`, and, when provided, `source_stage` and
`source_robot_id`. A recorded skeleton with SMPL-X joint semantics is not a
native SMPL-X parameter sequence. Reconstruction and post-RL robot motion are
separate source stages, even when their action ID is identical.

## Timing and object binding

Use the supplied timestamps as the authoritative clock. Public duration is
`timestamps_s[-1] - timestamps_s[0]`, not frame count divided by nominal rate.
The runner retains its nominal 30 Hz contract and existing rate tolerance.
Source object positions, rotations, meshes, and timestamps are validated before
registration and again before preparation. Object and body timestamps must
match. Missing hands or an absent object archive remain supported; explicitly
masked missing skeleton joints retain their existing semantics.

Selecting a different stage requires its corresponding objects, static scene
metadata, frame count, and clock. It must not inherit a reconstruction-stage
table pose or reuse a preview generated from the previous body source.

## Explicit source replacement

`python -m arenalab_workbench.source_migration --registry INPUT --plan PLAN
--output NEW_SNAPSHOT` prepares a new registry file and never activates it.
Each plan replacement names a recording ID, package directory, and expected
previous source SHA256. The package is validated using the regular source
attachment contract. Native previews must identify the exact replacement source
SHA256, robot, frame count, 30 Hz rate, and Z-up coordinates, and declare no
retargeting or trajectory edits. The replacement's declared end-effector mode
replaces the previous mode, including explicit `none`.

The replacement keeps source provenance and historical artifacts, removes the
old default scene/preview from the recording, and rebinds the source clock.
Sources with independent observations require a separate synchronization
migration; this command refuses to silently carry those observations across.
Activation must use a compare-and-swap against the live registry and preserve
private records and the public visibility allowlist.

## Jobs and cache identity

New jobs capture a source revision before dispatch. Deduplication requires the
same revision. The worker checks the revision before starting and before
publishing a default result under the registry lock. A changed source fails the
job explicitly and leaves its artifacts intact. A worker's provisional
completion is not exposed as a published success before this check.

Existing idempotency keys still identify their historical jobs. Source-aware
body caches include the selected source and object content identities; changing
the source does not overwrite immutable older priors.

Exports do not trust a catalog's saved scene URL as evidence of the current
source. A current source-aware scene key is accepted; a historical key requires
matching saved original-source SHA, source descriptor, object content and clock.
Decoder/refiner code hashes alone do not invalidate those historical exports.
Older complete-source receipts with independent observations must use the
current source key when their original observation binding cannot be verified.
Historical job links and already-created ZIPs remain available.

These checks establish representation, provenance, and clock consistency.
They do not certify physical feasibility, controller tracking, or geometric
contact accuracy.
