# Isolated test deployment HTTP QA

Host: http://motion-test.192-222-53-217.nip.io (test host; no production request or restart).

The live health and capabilities endpoints returned 200 / API 1.2.0. Workspace session creation returned 200. The public catalog returned 94 recordings, including exactly 10 HiPhi entries: all have null source video, provided full-body source, SMPL-X family, and no hand attachment. Their full IDs and source labels are in `recordings-sanitized.json`.

Only `/plans` was submitted, never `/runs`. HiPhi0079 with X2 requires an explicit scale decision (neutral-height estimate ratio 0.766); both preserve and match_robot plans are executable with their corresponding mode. G1 also requires a choice, while A3 defaults to preserve without a mismatch prompt. These are live backend preflight checks, not evidence of a visible dialog. The actual uniform scale is calculated separately from eight limb lengths.

All five JavaScript files served by the test host match the local built bytes, including dynamic page modules. See `all-javascript-bundles.json`. The source-body/source-object basis and FOV similarity functionality has CPU/Three.js tests documented in the parent validation directory.

Actual browser visual QA could not be performed: CUA reported `Browser is not available: iab`, and its surface inventory was `apps: [], browsers: []`. No screenshots were fabricated. Interactive selection, rendered geometry and no-RGB empty state still require a live browser check.

The captured plan response predates the subsequent API wording correction: supplied HiPhi must be described as `body_model: provided`, `source_body_family: smplx`, and `generated_body_model: null`, not native-SMPL-H reconstruction. The corrected code has regression coverage; confirm the new response after the safe API reload.
