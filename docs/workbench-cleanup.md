# Workbench cleanup — 2026-10-19

Production code now lives in `src/arenalab_workbench`, with a real package and
`arena-workbench` entry point. The two previous experiment source trees were
removed; there is no archive folder or compatibility import shim. The old
`body_frames` helper was migrated into the package and the duplicate SMPL-X
transfer was removed. Retired sources remain recoverable from Git history.

Removed chains: historical background fusion; complete-object mesh reconstruction,
tracking and validation; old WiLoR/video-hand recovery; unused generations of
contact estimation; obsolete `/umi-height`. Current contact/proximity workers,
DA3, SAM 3.1, Qwen, robot composites, height calibration and user dataset exports
remain. The frontend no longer loads or tracks obsolete meshes/surfaces. Current
semantic masks, segmented points and instance boxes continue to render.

`robot_registry`, `gripper_registry` and `height_registry` provide shared data
without pulling HTTP handlers into compute workers. Server paths and runtime
interpreter overrides live in `settings.py`. Scene jobs snapshot an importable
package with the correct parent on `PYTHONPATH`; no experiment `sys.path`
injection is needed. See [architecture](architecture.md) for the current flow
and [workbench guide](workbench-guide.md) for operator/developer details.

All maintained tests are in `tests/`. Script-style checks were made collectable;
old serial-orchestrator mocks and obsolete fixture assumptions were updated.
The source migration receipt and numerical comparisons are in
`validation/workbench-cleanup-20261019.json`. Ten core numerical modules have
identical ASTs after normalizing imports/docstrings. No model weights, sampling
rounds, conditioning points or refinement math were changed.

Validation: 96 Linux tests passed, with two private integration tests enabled
separately and both passing; 10 frontend tests, TypeScript/Vite build, wheel
packaging, CLI help and an isolated API factory smoke test passed. Full model
inference was not rerun for this structural change. The previous production
service and generated assets were left untouched during validation.

## Updating a deployment

Install this checkout in the relevant runtime or set `PYTHONPATH` to its `src`.
Use `python -m arenalab_workbench` or the `arena-workbench` console script as the
service entry point, replacing the old experiment-path command. Existing model,
source and store directories can stay in place using the documented settings.
The viewer code signature now identifies conversion dependencies; the first
switch to its new signature format can invalidate old unsigned/differently
signed viewer entries. Preserve existing output files; do not silently relabel
old results as new. Workbench human-prior and robot provenance checks remain.

No new performance speedup is claimed from this code cleanup. The previously
validated GPU JAX, exact frame reuse and SAM feature caching remain in place.
