# Isolated test deployment HTTP QA

Host: http://motion-test.192-222-53-217.nip.io (test host; no production request or restart).

The live health and capabilities endpoints returned 200 / API 1.2.0. Workspace session creation returned 200. The public catalog returned 94 recordings, including exactly 10 HiPhi entries: all have null source video, provided full-body source, SMPL-X family, and no hand attachment. Their full IDs and source labels are in `recordings-sanitized.json`.

Only `/plans` was submitted, never `/runs`. HiPhi0079 with X2 requires an explicit scale decision (neutral-height estimate ratio 0.766); both preserve and match_robot plans are executable with their corresponding mode. G1 also requires a choice, while A3 defaults to preserve without a mismatch prompt. These are live backend preflight checks, not evidence of a visible dialog. The actual uniform scale is calculated separately from eight limb lengths.

All five JavaScript files served by the test host match the local built bytes, including dynamic page modules. See `all-javascript-bundles.json`. The source-body/source-object basis and FOV similarity functionality has CPU/Three.js tests documented in the parent validation directory.

Actual browser visual QA could not be performed: CUA reported `Browser is not available: iab`, and its surface inventory was `apps: [], browsers: []`. No screenshots were fabricated. Interactive selection, rendered geometry and no-RGB empty state still require a live browser check.

The captured plan response predates the subsequent API wording correction: supplied HiPhi must be described as `body_model: provided`, `source_body_family: smplx`, and `generated_body_model: null`, not native-SMPL-H reconstruction. The corrected code has regression coverage; confirm the new response after the safe API reload.

## Generated A3/G1 assets

Both public scenes passed read-only HTTP asset validation. Each has 300 frames at nominal 30 Hz, raw source SMPL-X mesh (10,475 vertices / 20,908 faces), one Table_C_1 object, no source RGB, and source/object clocks that match exactly. Both 37,710,000-byte vertex streams and face/clock/object files match their manifest SHA256. Vertex/index ranges, finite motion data, GLB headers, all frame counts and serialized wrist errors were checked.

A3 preserve and G1 match_robot publish identical unscaled source geometry, with G1 applying a scene similarity of 0.8666085379738271 about the declared first-pelvis floor anchor. Across all 600 wrist samples, G1's exported numerical reference equals that same similarity applied to A3's original reference with max difference exactly 0 m. Thus numerical goals and displayed source geometry use the same transform, rather than scaling only the preview.

The saved independent native FK export audit covers all 300 frames. A3/G1 visual position error is 0; maximum rotation matrix discrepancy is 5.67e-15 / 5.22e-15; native joint discrepancy is 0; wrist task discrepancy is below 5e-16 m. Both immutable-prior checks are true. This checks exported kinematics against saved native outputs, not dynamics, reconstruction accuracy, or new optimization convergence. Both solver records explicitly report stopped/convergence unknown. See `generated-assets-qa.json`.
