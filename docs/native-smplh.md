# Native SMPL-H body reconstruction

The workbench keeps EgoAllo's native SMPL-H geometry and 52-joint skeleton throughout body export and robot-input preparation. The H-to-X conversion stage is removed. This change does not modify robot refinement objectives, weights, iteration budgets, contact policy, or other pipeline algorithms.

## Artifacts

- `egoallo_smplh.npz`: immutable reconstruction output, including native vertices, root/joint poses, local quaternions, fixed shape and source timestamps.
- `smplh_motion.npz`: native geometry and world-space skeleton for downstream body consumers.
- `canonical.npz`: native-H skeleton with declared basis/unit conversion for the existing robot encoder.
- `human_reference.json`: family, topology, model/source hashes and explicit no-transfer provenance.
- `human_validation.json`: native FK/world/local rotation checks, source wrist errors and source immutability status.

`emit_smplh.py` checks the 52-joint topology, fixed shape, composed joint rotations, forward kinematics and synchronized 30 Hz clock. Export preserves native vertices and world joint positions; the canonical encoder representation converts basis and metres to centimetres. Existing output artifacts are not overwritten.

## Cache migration

Native-H body policy identities use fresh versioned workspaces. Consumers reject retired converted-body manifests and descriptors, or a legacy converted motion cache without a native descriptor. Environment variables cannot select SMPL-X as a body-family fallback. Old recordings and historical results remain on disk as evidence; do not rename an old converted result to `smplh_motion.npz`.

Streaming export, body-library verification, upload dependencies, dataset exports and body-input readers use the same native-H contract. Mobile/gripper reader edits only adapt the human input and cache identity; their optimization algorithms are unchanged. Generic SDK dataset formats outside this workbench migration are unchanged.

## Validation

`python -m pytest tests/workbench -q -rs`

The portable regression fixture uses a translated, rotated, articulated 52-joint skeleton. It verifies vertex, world-pose, canonical-basis and timestamp preservation; rejects changing shape and converted caches; checks resumed window export without any X-model asset; and rejects partial immutable exports. No private recordings or licensed model weights are included.

Removing model conversion does not certify the reconstructed floor, contact, balance, dynamics, or real-robot execution. Native-H reconstruction can still penetrate a floor.
