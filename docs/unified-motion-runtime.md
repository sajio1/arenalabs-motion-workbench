# Unified motion runtime

The installed `arenalab_motion` package preserves the pre-fixed-hand-tolerance
HiPhi v3 recipe. Its provenance records the source hashes. It uses the same
60-step soft / 20-step polish limits, body and surface priors, wrist orientation
term, temporal regularization and native geometric floor handling for every
humanoid. No 5/8/10 mm experiment is exposed as an API option. Stop reasons and
measured errors remain separate from physical feasibility or WBC performance.

## Execution and data boundary

The request runner is `python -m arenalab_motion.runner --request request.json`.
It consumes explicit local source, model/runtime and robot asset paths and writes
immutable raw-prior receipts and a separate refined result. It does not upload a
URDF or fetch a model. The deployment supplies cached weights and model files.

The numerical stages use resident CUDA tensors, analytic/autodiff kinematics,
batched constraint assembly, GPU FP64 interior-point QP steps and block cyclic
reduction. Source file decoding, body mesh materialization, asset loading,
serialization and visualization export still have CPU work. A single trajectory
is solved on one GPU; independent jobs/stages are distributed across the selected
GPU pool. This is job parallelism, not distributed single-trajectory training or
an implemented cross-host tensor transport protocol.

For a customer that cannot share a URDF, run this package and target decoder on
their own endpoint. Source motion can be supplied to that endpoint without
returning robot assets. The current API server with robot import is an alternative
self-hosted deployment, not a requirement to upload proprietary robot geometry to
ArenaLabs. An authenticated cloud-to-customer worker queue and privacy-reviewed
latent-only wire protocol are not included in this release; do not claim those
services are deployed merely because the numerical runner is local.

## Samples and release validation

Git contains the ten-entry HiPhi manifest, provenance metadata and registration
tool only. Actual motion arrays and object meshes are staged separately on the
test server, outside both Git and the HTTP asset root. Licensed SMPL body models
and neural weights are also private runtime dependencies. HiPhi samples are
labelled as the existing GPU-converted SMPL-X inputs; generated bodies remain
native SMPL-H.

Same-machine migration checks cover a full 300-frame G1 clip: immutable source
codes, raw decoder output, native initialization, final qpos and operation targets
match the research baseline exactly. Independent native FK differs by at most
1.34e-15 m. A 16-frame native-H input also passes the actual bridge/CLI/export
contract. These are port/parity checks, not evidence of improved model quality,
new embodiment generalization or WBC execution.

The Lambda test instance uses separate store, job roots, API database and service.
It shares the production manipulation GPU lease directory and checks idle memory
before allocation. Older production locomotion workers do not honor that lease;
avoid launching those concurrently with a full four-card test. This release does
not modify the production service to retrofit its scheduler.

Merge destination is `codex/platform-api-v1`. Updating `main` and replacing the
production deployment require the user's later test decision.
