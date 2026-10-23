"""Lossless local SDK archive -> named native robot states, checked by full FK.

This changes storage convention only. It performs no inference or retargeting.
Source model paths come from the decoding endpoint's trusted robot registry.
"""
from pathlib import Path
import hashlib
import numpy as np


def convert(source, row, destination):
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from alphamotion.engine.descriptor import build_from_mjcf
    source, destination = Path(source), Path(destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as z:
        q, root, rot6d = z['q'], z['root_t'], z['rot6d']
        names = z['joint_names'].tolist()
        expected = z['world_position_cm'] / 100
        fps = float(z['fps'])
    if fps != 30 or any(not np.isfinite(v).all() for v in (q, root, rot6d, expected)):
        raise ValueError('Expected finite, explicitly sampled 30 Hz original robot states')
    spec, _, _, bindings, _ = build_from_mjcf(row['xml'], row['body'])
    model = mj.MjModel.from_xml_path(row['xml']); data = mj.MjData(model)
    joints = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
    values = {name: q[:, names.index(spec.joint_names[j]), axis]
              for j, group in enumerate(bindings) for axis, name in enumerate(group)}
    if set(values) != set(joints):
        raise ValueError('Archive must cover every native source hinge exactly')
    C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    r = rot6d[:, 0]
    x = r[:, :3] / np.linalg.norm(r[:, :3], axis=1, keepdims=True)
    y = r[:, 3:] - (x * r[:, 3:]).sum(1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    R = C.T @ np.stack((x, y, np.cross(x, y)), axis=-1) @ C
    arrays = dict(joint_names=np.asarray(joints), joint_positions=np.stack([values[j] for j in joints], -1),
                  root_position=root / 100 @ C,
                  root_quaternion_wxyz=Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]],
                  timestamps_s=np.arange(len(q)) / fps)
    ids = [model.body(n).id for n in names]
    errors = []
    for t in range(len(q)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = arrays['root_position'][t]
        data.qpos[3:7] = arrays['root_quaternion_wxyz'][t]
        for j in joints: data.qpos[model.jnt_qposadr[model.joint(j).id]] = values[j][t]
        mj.mj_forward(model, data)
        errors.append(np.linalg.norm(data.xpos[ids] @ C.T - expected[t], axis=-1))
    errors = np.asarray(errors)
    if not np.isfinite(errors).all() or errors.max() > 1e-6:
        raise ValueError('Original full-body FK mismatch: ' + str(errors.max()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as z:
            if set(z.files) != set(arrays) or any(not np.array_equal(z[k], v) for k, v in arrays.items()):
                raise FileExistsError('Immutable named source already exists with different contents')
    else:
        with destination.open('xb') as stream: np.savez_compressed(stream, **arrays)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Original archive changed during conversion')
    return dict(source_path=str(source), source_sha256=digest, output_path=str(destination),
                output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                frames=len(q), fps=fps, hinges=len(joints), fk_checked_bodies=len(ids),
                fk_error_mean_m=float(errors.mean()), fk_error_p95_m=float(np.quantile(errors,.95)),
                fk_error_p99_m=float(np.quantile(errors,.99)), fk_error_max_m=float(errors.max()),
                resampled=False, human_model=None, source_trajectory_modified=False)
