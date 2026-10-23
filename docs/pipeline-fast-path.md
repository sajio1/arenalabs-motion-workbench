# Pipeline fast paths

The manipulation browser requests a 50-record page and filters on the server.
A `?clip=` link resolves one authorized recording; a `?scene=` link loads that
scene directly. POST `/api/v1/sessions` now returns `{ "ready": true }` and only
creates the workspace cookie. Consumers must use `/recordings` for catalogue
contents. Existing workspace cookies, ownership and idempotency remain unchanged.

Display models are published by content hash under `manipulation4d/robot-models`.
Each owner has a separate namespace; private models remain `no-store` and require
the workspace cookie. Gzip transport is lossless and does not change mesh vertices,
joint bindings or motion. Per-result model files remain present for portable
exports. To prepare historical public models, run:

```
python -m arenalab_workbench.model_assets --store /path/to/store
```

This creates an internal alias index; it does not rewrite historical scene or
trajectory files. File stat changes invalidate an alias. The browser retains up
to three parsed target models, pins displayed geometry, and owns transforms per
instance. Rejected loads can be retried. Playback JSON is only deduplicated briefly
because progressive results may change. Catalogue pagination uses server offsets,
not the length of deduplicated visible rows.

The native solver defaults to zero resource cooldown. All iteration limits,
objectives, constraints, numerical tolerances and export checks are unchanged.
GPU workers are started lazily under the existing per-card lease. They retain one
Prime codec and bounded static target/geometry caches across requests. The cache
keys include the complete robot row and XML/mesh dependency hashes. Fresh MuJoCo
data is allocated per job; motion, targets and solver states are never reused.
All robot geometry remains on the machine executing the endpoint.

Workers communicate over a private Unix socket, not a network listener. Each job
passes its real scheduler flock descriptor with SCM_RIGHTS, so an interrupted
caller does not expose a busy GPU. Workers release the descriptor after finishing,
exit on numerical/runtime failure, and expire after 30 idle minutes. No failed
job is silently replayed. A different source release starts a different worker.
The scheduler recognizes only its own idle process and still rejects cards with
external workloads. Idle workers can be evicted when another stage needs memory.

`ARENA_RESIDENT_MOTION=0` disables persistent workers while retaining the same
numerical algorithm. Standalone CLI calls without a scheduler lease continue in
a separate process. Request/attempt files and failure evidence stay immutable.
Timing reports distinguish `model_cache_hit`, `geometry_cache_hit`,
`refinement_active_s`, `refinement_wall_s` and total runner wall time.
