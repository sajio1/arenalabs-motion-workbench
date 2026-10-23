# Output selection validation

API 1.3 adds requested trajectory, point-cloud, contact and inpaint/FOV modalities. This release changes routing, publication and caching; the resident skeleton/surface/wrist numerical policy is unchanged.

- Full Python suite: 268 passed, 2 skipped, 2 baseline failures. The two `test_scene_export` failures require a missing `source` in the historical fixture; they were reproduced on original branch commit `b3eeeee9ebb7266e7cae5a1f93c64844951b987f` (see `../unified-release/baseline-scene-failures.txt`).
- Run with `PYTHONPATH=src ARENA_SCHEDULER_ROOT=/tmp/arena-output-selector-scheduler` and the local Python 3.12 runtime. Scheduler isolation is required on hosts without `/home/ubuntu`.
- Stage tests mock model execution and assert requested/dependency dispatch, independent contact observation preparation, result filtering, cache reuse and explicit missing-artifact failure. They are orchestration tests, not model-quality evidence.
- Source capabilities inspect registered camera/calibration and actual observed hand streams. Public preview video alone never enables CV.
- Actual source datasets and weights remain outside Git on Lambda. No main deployment is part of this release.
- Browser rendering could not be visually inspected because no computer-use browser surface was available. HTTP/static checks and frontend unit/build checks do not replace visual acceptance.

Final frontend verification: 43 Node tests passed; TypeScript and Vite build passed. API/SDK/viewer targeted suite: 61 tests passed. See `API_WEB_QA.md` for scope.
