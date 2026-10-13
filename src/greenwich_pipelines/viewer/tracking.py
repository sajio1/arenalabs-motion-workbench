"""Measure final native FK against declared wrist task frames; never fit motion."""
import json
from pathlib import Path
import numpy as np

C = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])


def wrist_metrics(actual_p, target_p, actual_r, target_r):
    arrays = [np.asarray(v, dtype=float) for v in (actual_p, target_p, actual_r, target_r)]
    actual_p, target_p, actual_r, target_r = arrays
    if (actual_p.shape != target_p.shape or actual_p.ndim != 3 or actual_p.shape[1:] != (2, 3)
            or actual_r.shape != target_r.shape or actual_r.shape != (*actual_p.shape[:2], 3, 3)
            or not all(np.isfinite(v).all() for v in arrays)):
        raise ValueError('Invalid paired wrist tracking samples')
    error = np.linalg.norm(actual_p - target_p, axis=-1) * 1000
    cosine = (np.einsum('...ij,...ij->...', actual_r, target_r) - 1) / 2
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    return dict(schema='arenalab.tracking.v1', roles=['left_wrist', 'right_wrist'],
                reference='Input task targets vs final native FK', unit='m', axis='y-up',
                actual_positions=actual_p.tolist(), target_positions=target_p.tolist(),
                position_error_mm=error.tolist(), orientation_error_deg=angle.tolist(),
                thresholds=dict(position_mm=10, orientation_deg=15))


def native_descriptor_wrists(row, saved, model):
    import mujoco as mj
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.embodiment.mjcf_build import build_merge
    spec, _, _, qnames, _ = build_from_mjcf(row['xml'], row['body'])
    if list(spec.joint_names) != saved['joint_names'].tolist():
        raise ValueError('Tracking model differs from saved descriptor')
    keep, frames = build_merge(model)
    indices = saved['endpoint_indices'][:2].astype(int)
    body_ids = [keep[i] for i in indices]
    frame_ids = [frames[i] for i in body_ids]
    root = keep[0]
    slots = []
    for j, names in enumerate(qnames):
        for axis, name in enumerate(names):
            joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE:
                raise ValueError('Unsupported tracking motor binding')
            slots.append((j, axis, int(model.jnt_qposadr[joint])))
    a = saved['rot6d'][:, 0]
    x = a[:, :3] / np.linalg.norm(a[:, :3], axis=-1, keepdims=True)
    y = a[:, 3:] - np.sum(x * a[:, 3:], axis=-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    roots = C.T @ np.stack((x, y, np.cross(x, y)), -1) @ C
    translations = saved['root_t'] @ C / 100
    data = mj.MjData(model)
    positions, rotations = [], []
    for t, q in enumerate(saved['q']):
        data.qpos[:] = model.qpos0
        for j, axis, address in slots:
            data.qpos[address] = q[j, axis]
        mj.mj_forward(model, data)
        delta = roots[t] @ data.xmat[frames[root]].reshape(3, 3).T
        positions.append((translations[t] + (data.xpos[body_ids] - data.xpos[root]) @ delta.T) @ C.T)
        rotations.append(C @ (delta @ data.xmat[frame_ids].reshape(2, 3, 3)) @ C.T)
    return np.asarray(positions), np.asarray(rotations)


def tracking_payload(row, saved, pipeline, model):
    if pipeline == 'nymeria':
        actual_p, actual_r = native_descriptor_wrists(row, saved, model)
        return wrist_metrics(actual_p, saved['input_positions_yup_cm'][:, :2] / 100,
                             actual_r, saved['target_rotation_yup'][:, :2])
    if pipeline == 'gento':
        import mujoco as mj
        metadata = json.loads(Path(row['metadata']).read_text())
        mounts = {f['link']: f for f in metadata['end_effector_frames']}
        ids = saved.get('wrist_ids', [model.body(name).id for name in metadata['wrists']])
        local_p = saved['ee_local_position']
        local_r = np.asarray(saved.get('semantic_local', [mounts[name]['rotation'] for name in metadata['wrists']]))
        data = mj.MjData(model)
        positions, rotations = [], []
        for q in saved['qpos']:
            data.qpos[:] = q
            mj.mj_forward(model, data)
            r = data.xmat[ids].reshape(2, 3, 3)
            positions.append((data.xpos[ids] + np.einsum('kij,kj->ki', r, local_p)) @ C.T)
            rotations.append(C @ r @ local_r)
        return wrist_metrics(positions, saved['targets'] @ C.T, rotations, C @ saved['target_orientation'])
    return None  # Whole-body source joints are not a declared two-wrist task contract.
