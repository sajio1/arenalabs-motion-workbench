# Complete robot motion input

Complete robot joint recordings enter the existing AlphaMotion and
skeleton/surface/wrist refiner directly. They do not use a human model or
three-point reconstruction. Routing depends on the declared input format,
not dataset names. Existing human-body and sparse-observation routes remain.

Attach a `motion_source` to the recording manifest:

```json
{
  "format": "robot_joint_state",
  "family": "robot",
  "robot_id": "g1",
  "path": "joint-state.npz",
  "world_up": "z",
  "units": "m",
  "floor_z_m": 0.0,
  "evidence": "simulation_fk"
}
```

The NPZ supplies `joint_names`, `joint_positions` in radians, `root_position`
in metres, `root_quaternion_wxyz` and `timestamps_s`. All named source hinge
joints are required, with a floating root and a contiguous 30 Hz clock.
The source robot ID resolves an installed local descriptor. Uploads cannot
select an arbitrary model path. Invalid complete inputs fail instead of
falling back to human reconstruction.

The encoder receives every descriptor node and its native DoF features,
topology, positions and orientations. Task frames and limb directions are
separate views of the original full FK. Independent recorded TCP targets may
be supplied separately; synthesized head/wrist proxies are not needed for a
complete robot recording. `preserve` and `match_robot` retain their existing
meaning; the latter fits the source robot's eight limb segments to the target.

`tools/native_archive_to_robot_state.py` converts an existing SDK archive to
named native state without fitting or resampling, and independently checks
every archived body position against local MuJoCo FK. Keep both artifacts and
their provenance. `attach_robot_motion` prepares a registry snapshot without
activating it. Existing GRAIL inputs must be reattached to their original G1
states; relabelling sparse or reconstructed caches is insufficient.

Complete robot scene identities include the attachment, so historical human
reconstructions are not reused as fresh robot results. Preview assets contain
one static robot mesh plus exact per-frame rigid-link poses. No dense
per-frame vertex movie is generated. The viewer receives `source_robot`;
human sources continue to use `source_body`.

Validation: named joint reordering, complete FK coverage, invalid states,
clock/rotation preservation, explicit scaling, preview source identity,
scene cache separation, and existing supplied-human/API regression tests.
This input change does not certify dynamics, WBC execution, object contact,
or self-collision, and it does not establish a 5× end-to-end latency bound.
