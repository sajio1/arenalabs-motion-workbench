"""Export EgoAllo's native H geometry and skeleton without body-model transfer."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from greenwich_motion_sdk.motion import MotionClip
from arenalab_workbench.human_reference import SCHEMA, SMPLH_POLICY

BODY_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee',
              'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
              'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar',
              'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
              'right_elbow', 'left_wrist', 'right_wrist']
NAMES = BODY_NAMES + [f'{side}_{finger}{joint}' for side in ('left', 'right')
                     for finger in ('index', 'middle', 'pinky', 'ring', 'thumb')
                     for joint in (1, 2, 3)]
ALIASES = {'pelvis': 'Hips', 'head': 'Head', 'left_hip': 'LeftLeg',
           'right_hip': 'RightLeg', 'left_knee': 'LeftShin', 'right_knee': 'RightShin',
           'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot',
           'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase',
           'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm',
           'left_elbow': 'LeftForeArm', 'right_elbow': 'RightForeArm',
           'left_wrist': 'LeftHand', 'right_wrist': 'RightHand'}


def canonical_names():
    result=[]
    for name in NAMES:
        if name in ALIASES:result.append(ALIASES[name]);continue
        side,_,distal=name.partition('_')
        if side in ('left','right') and distal.startswith(('index','middle','pinky','ring','thumb')):
            result.append(side.title()+'Hand'+distal.title())
        else:result.append(''.join(part.title() for part in name.split('_')))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(root):
    root = Path(root)
    destinations = ['canonical.npz', 'smplh_motion.npz', 'human_reference.json',
                    'human_validation.json', 'sources.json']
    if any((root / name).exists() for name in destinations):
        raise FileExistsError('Native H exports are immutable; use a fresh body workspace')
    raw_path = root / 'egoallo_smplh.npz'
    raw_hash = digest(raw_path)
    with np.load(raw_path) as saved:
        d = {key: saved[key] for key in saved.files}
    with np.load(root / 'model.npz', allow_pickle=True) as saved:
        model = {key: saved[key] for key in saved.files}
    beta = np.asarray(d['betas'][0], dtype=np.float32)
    if not np.allclose(d['betas'], beta, atol=1e-5, rtol=0):
        raise ValueError('Native H export requires the fixed-shape EgoAllo policy')
    if 'vertices' not in d:
        raise ValueError('Native H vertices are required; do not synthesize X geometry')
    n = len(d['root_pose'])
    parents = np.r_[-1, np.asarray(model['kintree_table'][0, 1:], dtype=int)]
    if len(parents) != 52 or any(p >= j for j, p in enumerate(parents)):
        raise ValueError('Unexpected native H skeleton topology')
    neutral = np.asarray(model['v_template'], dtype=np.float32) + np.einsum(
        'vck,k->vc', np.asarray(model['shapedirs'][..., :len(beta)], dtype=np.float32), beta)
    rest = np.asarray(model['J_regressor'], dtype=np.float32) @ neutral
    rest = rest - rest[:1]
    offsets = np.zeros_like(rest)
    offsets[1:] = rest[1:] - rest[parents[1:]]
    world_pose = np.concatenate([d['root_pose'][:, None], d['joint_poses']], axis=1)
    world = world_pose[..., 4:7].copy()
    global_rotation = Rotation.from_quat(
        world_pose[..., :4].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(n, 52, 3, 3)
    local_q = np.concatenate([d['root_pose'][:, None, :4], d['local_quats']], axis=1)
    local_rotation = Rotation.from_quat(local_q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(n, 52, 3, 3)
    fk = np.zeros_like(world, dtype=np.float64)
    fk[:, 0] = world[:, 0]
    rotation_error = 0.
    for j in range(1, 52):
        parent = parents[j]
        fk[:, j] = fk[:, parent] + np.einsum('tij,j->ti', global_rotation[:, parent], offsets[j])
        rotation_error = max(rotation_error, float(np.max(np.abs(
            global_rotation[:, parent] @ local_rotation[:, j] - global_rotation[:, j]))))
    fk_error = float(np.max(np.linalg.norm(fk - world, axis=-1)))
    if fk_error > 2e-5 or rotation_error > 2e-5:
        raise ValueError(f'Native H FK contract failed: position={fk_error}, rotation={rotation_error}')
    times = d['timestamps_s']
    if len(times) != n or (n > 1 and not np.allclose(np.diff(times), 1/30, atol=1e-6)):
        raise ValueError('Native H export supports synchronized 30 Hz only')
    vertices = d['vertices']
    if vertices.shape != (n, len(neutral), 3):
        raise ValueError('H vertex topology differs from source model')
    if not all(np.isfinite(a).all() for a in (vertices, world, global_rotation, beta)):
        raise ValueError('Non-finite native H artifact')
    basis = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    provenance = dict(source='EgoAllo head/wrist reconstruction', human_model='SMPL-H',
        export_policy=SMPLH_POLICY, transfer='none; native geometry and joint poses preserved',
        source_sha256=raw_hash, fps=30, fingers='Unobserved body-prior pose; no finger-action generation')
    MotionClip(canonical_names(), parents, offsets * 100, basis @ global_rotation,
               world @ basis.T * 100, times, provenance).save(root / 'canonical.npz')
    np.savez_compressed(root / 'smplh_motion.npz', vertices=vertices, faces=d['faces'],
        joints=world, betas=beta, global_rotation=global_rotation,
        local_axis_angle=Rotation.from_matrix(local_rotation.reshape(-1, 3, 3)).as_rotvec().reshape(n, 52, 3),
        root_translation=world[:, 0], timestamps_s=times, human_model=np.array('smplh'),
        joint_names=np.asarray(NAMES), parents=parents)
    manifest = json.loads((root / 'input_manifest.json').read_text())
    height = ((rest[[7, 10, 8, 11], 1] - (neutral[:, 1] -
              (np.asarray(model['J_regressor'], dtype=np.float32) @ neutral)[0, 1]).min()) * 100).tolist()
    source = dict(id='nymeria_' + manifest.get('clip', 'motion'), format='canonical', path='canonical.npz',
        roles=['Hips', 'Head', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'],
        foot_pairs=[['LeftFoot', 'LeftToeBase'], ['RightFoot', 'RightToeBase']],
        foot_landmark_heights_cm=height)
    (root / 'sources.json').write_text(json.dumps(dict(schema='alphamotion.sources.v1', sources=[source]), indent=2))
    descriptor = dict(schema=SCHEMA, family='smplh', policy=SMPLH_POLICY,
        motion='smplh_motion.npz', model='model.npz', joint_count=52, vertex_count=len(neutral),
        source='egoallo_smplh.npz', source_sha256=raw_hash,
        model_sha256=digest(root / 'model.npz'), geometry_transfer=False)
    (root / 'human_reference.json').write_text(json.dumps(descriptor, indent=2))
    obs = np.load(root / 'observations.npz')
    error = np.linalg.norm(world[:, [20, 21]] - obs['wrist_position'], axis=-1) * 1000
    report = dict(frames=n, human_model='SMPL-H', export_policy=SMPLH_POLICY,
        transfer='none', native_joint_export_max_error_mm=0., native_vertex_export_max_error_mm=0.,
        skeleton_fk_max_error_mm=fk_error * 1000, global_rotation_composition_max_error=rotation_error,
        source_wrist_error_mm=dict(mean=float(error.mean()), p95=float(np.percentile(error, 95)), max=float(error.max())),
        source_sha256=raw_hash, source_unchanged=digest(raw_path) == raw_hash,
        fixed_shape=True, physical_validation=False)
    (root / 'human_validation.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    print(json.dumps(export(Path(os.environ['ARENA_EGO_WORKDIR'])), indent=2))
