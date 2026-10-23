# Portable native reference data

Every newly completed robot dataset uses one versioned output contract,
regardless of whether its source was human motion, three tracked points or
recorded robot data. No human body-model decoder is needed to consume it.

After extracting the ZIP, verify the entire package:

```sh
python validate_dataset.py /path/to/extracted-dataset
```

The offline validator requires NumPy and PyArrow. It verifies the exact file
inventory and SHA256s, all frame clocks, native and LeRobot named joint tables,
all attachment channels and selected modality paths. Editing or adding files
inside the immutable dataset invalidates its integrity check; put validation
reports and simulator outputs elsewhere.

Use the native reference folder and your matching local robot model:

```sh
cd /path/to/extracted-dataset/extensions/arenalabs/native_reference
python native_reference_mujoco.py --bundle . --model /local/robot/model.xml --validate
python native_reference_mujoco.py --bundle . --model /local/robot/model.xml
```

With Isaac Sim's Python:

```sh
/path/to/isaac/python.sh native_reference_isaac.py --bundle . \
  --urdf /local/robot/robot.urdf --headless --validate \
  --output /tmp/isaac-reference-validation
```

`--usd` accepts an existing matching Z-up, metre-scale articulation instead of
URDF. An explicit `--joint-map` maps source names to local Isaac DOF names.
Missing DOFs, a different base link or inconsistent FK are errors. Both adapters
set articulated reference states. They do not run a locomotion/WBC controller.

The data contract fixes these semantics:

| Field | Meaning |
|---|---|
| `timestamp_s` | Exact relative 30 Hz replay clock, starts at zero |
| `source_timestamp_s` | Original source clock, retained separately |
| `root_position_m`, `root_quaternion_wxyz` | Native root link in original right-handed Z-up world |
| `joint_names`, `joint_position` | Named native body coordinates; per-joint hinge/slide units |
| `additional_joint_names`, `additional_joint_position` | Independently solved attached-hand/gripper coordinates; empty when absent |
| `native_qpos` | Complete MuJoCo reference state, addresses explicitly declared |
| `native_qvel` and named velocities | Derived reference derivatives, not recorded telemetry |
| `link_pose_world` | Named native body-link poses for independent FK comparison |

The derivative package preserves physical poses while making quaternion
hemispheres continuous. Original motion NPZ files and their hashes remain
separate. Never apply the web viewer's Y-up transform to the native reference,
and never treat LeRobot's next-frame `action` as a native simulator qpos vector.

Replay defaults to the body. A missing hand model is valid and does not fail
the run; available hand/gripper channels remain preserved in the package. Use
`--with-hands` to apply them with a matching local assembly. Only that explicit
mode requires the corresponding attachment names and model. Reports list the
mode and applied/optional coordinates. `--body-only` remains an explicit alias
for the default. The body FK receipt does not claim that attached-hand physics
or a controller were executed.

The adapters are local consumers: they do not upload URDF/MJCF, meshes or
credentials. The dataset still includes its previously selected visual assets.
Model topology/axes/reference transforms must match; a filename or equal number
of joints is insufficient. Physics readiness additionally requires the client's
collision/inertia/actuator model, control interface and controller validation.
