"""Supplied SMPL-family motion -> immutable body22/surface adapter.

This module never completes a supplied body, fits shape, smooths motion, changes
its clock or runs a robot solver. Generated EgoAllo bodies remain native SMPL-H.
External SMPL/X remain honestly labelled external SMPL/X; they are not renamed H.
Licensed body models are resolved from trusted endpoint configuration, not uploads.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'arenalab.motion-source.v1'
POLICY = 'supplied-body22-surface-v1'
C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
NAMES = ['Hips', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftShin', 'RightShin',
         'Spine2', 'LeftFoot', 'RightFoot', 'Chest', 'LeftToeBase', 'RightToeBase',
         'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm',
         'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
ROLES = np.array([0, 15, 20, 21, 7, 8])
FAMILY_JOINTS = {'smpl': 24, 'smplh': 52, 'smplx': 55}
MAX_NPZ_BYTES = 2 * 1024**3


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def _npz(path, keys=None):
    with zipfile.ZipFile(path) as archive:
        if sum(m.file_size for m in archive.infolist()) > MAX_NPZ_BYTES:
            raise ValueError('Source NPZ expanded size exceeds 2 GiB')
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files if keys is None or k in keys}


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite {name}')
    return value


def _unit(value):
    size = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(size < 1e-10):
        raise ValueError('Degenerate semantic direction in supplied body')
    return value / size


def _rotation(value, name):
    value = _finite(value, name)
    if value.shape[-2:] != (3, 3) or not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), atol=2e-5) or not np.allclose(np.linalg.det(value), 1., atol=2e-5):
        raise ValueError(f'{name} requires proper rotation matrices')
    return value


def source_route(manifest):
    """Route by supplied evidence, never by dataset/robot/action names."""
    if manifest.get('motion_source') is not None:
        spec = manifest['motion_source']
        from .robot_motion_source import is_robot
        if is_robot(spec):
            if not spec.get('robot_id'):raise ValueError('Source robot_id is required')
            return dict(route='supplied-robot', source_completion=False, family='robot', generated_human_model=None)
        if not isinstance(spec, dict) or spec.get('format') not in {'smpl_parameters', 'smpl_decoded'}:
            raise ValueError('Invalid supplied motion_source; it must not fall back to body inference')
        if spec.get('family') not in FAMILY_JOINTS:
            raise ValueError('Declare supplied family smpl, smplh or smplx')
        return dict(route='supplied-body', source_completion=False,
                    family=spec['family'], generated_human_model=None)
    return dict(route='three-point-egoallo', source_completion=True,
                family='smplh', generated_human_model='smplh',
                required_observations=['head pose', 'left wrist pose', 'right wrist pose'])


def validate_body_attachment(package, manifest):
    """Validate a portable upload attachment without opening server model paths.

    Returns a trusted absolute source descriptor for registry storage. Model
    assets are configured separately by the endpoint; uploads cannot select an
    arbitrary local file, Python module, pickle, command or network URL.
    """
    from .dataset_io.storage import within
    route = source_route(manifest)
    if route['route'] == 'supplied-robot':
        from .robot_motion_source import validate_attachment
        return validate_attachment(package,manifest)
    if route['route'] != 'supplied-body':
        return None
    spec = dict(manifest['motion_source'])
    forbidden = {'model_path', 'model', 'model_url', 'decoder', 'command'} & spec.keys()
    if forbidden:
        raise ValueError('Body models/decoders are endpoint configuration, not upload fields')
    path = within(package, spec.get('path', ''))
    if not path.is_file() or path.suffix.lower() != '.npz':
        raise ValueError('motion_source.path must identify an NPZ inside the recording package')
    if spec.get('world_up') not in {'y', 'z'} or spec.get('units') != 'm':
        raise ValueError('Declare body world_up=y/z and units=m')
    if spec.get('evidence') not in {'measured', 'fitted', 'reconstructed', 'synthetic'}:
        raise ValueError('Declare body evidence without treating fitted joints as measurements')
    if spec.get('objects_path'):
        object_path = within(package, spec['objects_path'])
        if not object_path.is_file() or object_path.suffix.lower() != '.npz':
            raise ValueError('objects_path must be an NPZ inside the recording package')
        spec['objects_path'] = str(object_path)
    floor = float(spec.get('floor_z_m', manifest.get('floor_z_m', float('nan'))))
    if not np.isfinite(floor):
        raise ValueError('Declare a finite fixed source floor_z_m')
    spec['floor_z_m'] = floor
    values = _npz(path)
    _validate_family(values, spec['family'])
    times, indices, _ = _clock(values, spec)
    if len(times) != manifest.get('frames', len(times)):
        raise ValueError('Supplied body frame count differs from recording')
    if 'timestamps_s' in values and len(indices) != len(values['timestamps_s']):
        raise ValueError('Upload body must already match the recording window; pre-slice locally')
    spec.update(path=str(path), sha256=sha256(path))
    return spec


def _validate_family(raw, family):
    declared = raw.get('human_model', raw.get('model_type', raw.get('model')))
    if declared is not None and np.asarray(declared).shape == () and str(declared).lower().replace('-', '') != family:
        raise ValueError('Supplied body family disagrees with payload; no H/X relabelling')


def _clock(raw, spec):
    if spec.get('format') == 'robot_joint_state':
        n = len(raw.get('joint_positions', []))
    else:
        n = len(raw['joints']) if 'joints' in raw and spec['format'] == 'smpl_decoded' else len(raw.get('poses', raw.get('local_axis_angle', [])))
    if n < 2:
        raise ValueError('At least two complete supplied body frames required')
    if 'timestamps_s' in raw:
        full = _finite(raw['timestamps_s'], 'timestamps_s')
        if full.shape != (n,):
            raise ValueError('Body timestamps must match the pose frames')
        dt = float(np.median(np.diff(full)))
    else:
        rate = float(raw.get('mocap_framerate', raw.get('fps', spec.get('fps', 0.))))
        dt = float(raw['frame_time']) if 'frame_time' in raw else (1. / rate if rate > 0 else float('nan'))
        full = np.arange(n, dtype=np.float64) * dt
    if not np.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(full), dt, atol=1e-7, rtol=1e-5):
        raise ValueError('Explicit uniform source clock required')
    for key in ['stride', 'start_frame', 'frames']:
        if key in spec and (isinstance(spec[key], bool) or not isinstance(spec[key], (int, np.integer))):
            raise ValueError('Frame selection requires exact integer ' + key)
    stride = int(spec.get('stride', 1)); start = int(spec.get('start_frame', 0))
    if stride < 1 or start < 0:
        raise ValueError('Positive stride and nonnegative start_frame required')
    count = int(spec.get('frames', (n - start + stride - 1) // stride))
    take = start + np.arange(count, dtype=np.int64) * stride
    if count < 2 or take[-1] >= n:
        raise ValueError('Source window outside the recording')
    fps = 1. / (dt * stride)
    if abs(fps - 30.) > .001:
        raise ValueError('Only nominal 30 Hz integer-stride input is supported; no implicit resampling')
    original = np.asarray(raw.get('source_frame_indices', np.arange(n)))
    if not np.issubdtype(original.dtype, np.integer):
        raise ValueError('Original source frame indices must be integers')
    original = original.astype(np.int64)
    if original.shape != (n,) or np.any(np.diff(original) <= 0):
        raise ValueError('Source frame indices must be strictly increasing')
    return full[take], original[take], take


def _model(path, family, beta):
    m = _npz(path, {'v_template', 'shapedirs', 'weights', 'J_regressor', 'kintree_table', 'posedirs', 'f', 'faces'})
    count = FAMILY_JOINTS[family]
    vertices = _finite(m['v_template'], 'v_template')
    shape = _finite(m['shapedirs'], 'shapedirs')
    if vertices.ndim != 2 or vertices.shape[1] != 3 or shape.shape[:2] != vertices.shape or shape.shape[2] < len(beta):
        raise ValueError('Licensed model shape basis does not match supplied betas')
    weights = _finite(m['weights'], 'weights')
    regressor = _finite(m['J_regressor'], 'J_regressor')
    if weights.shape != (len(vertices), count) or regressor.shape != (count, len(vertices)):
        raise ValueError('Licensed model topology does not match declared SMPL family')
    parents = np.r_[-1, np.asarray(m['kintree_table'][0, 1:], dtype=np.int64)]
    if parents.shape != (count,) or not np.array_equal(parents[:22], PARENTS) or any(p < 0 or p >= j for j, p in enumerate(parents[1:], 1)):
        raise ValueError('Unexpected SMPL body topology')
    if np.any(weights < -1e-7) or not np.allclose(weights.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid model skinning weights')
    shaped = vertices + np.einsum('vck,k->vc', shape[..., :len(beta)], beta)
    rest = regressor @ shaped
    faces = np.asarray(m.get('f', m.get('faces')), dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid model triangle topology')
    return m, shaped, rest, parents, weights, faces


def _beta(raw):
    beta = _finite(raw['betas'], 'betas')
    if beta.ndim == 2:
        if not np.allclose(beta, beta[:1], atol=1e-6, rtol=0):
            raise ValueError('Changing per-frame shape requires a separately versioned source adapter')
        beta = beta[0]
    if beta.ndim != 1 or not len(beta):
        raise ValueError('Fixed betas vector required')
    return beta


def _decode(raw, spec, take, model_path):
    family = spec['family']; beta = _beta(raw)
    model, shaped, rest, parents, weights, faces = _model(model_path, family, beta)
    count = len(parents)
    if spec['format'] == 'smpl_decoded':
        joints = _finite(raw['joints'], 'joints')[take]
        rotation = _rotation(raw['global_rotation'], 'global_rotation')[take]
        vertices = _finite(raw['vertices'], 'vertices')[take]
        if joints.shape != (len(take), count, 3) or rotation.shape != (len(take), count, 3, 3) or vertices.shape != (len(take), len(shaped), 3):
            raise ValueError('Decoded motion topology does not match the actual body model')
        if 'faces' in raw and not np.array_equal(raw['faces'], faces):
            raise ValueError('Decoded source and body model triangle topology disagree')
        # Decoded matrices map native model Y-up local axes to declared world.
        fk = joints[:, :1].repeat(count, 1)
        for j in range(1, count):
            fk[:, j] = fk[:, parents[j]] + np.einsum('tij,j->ti', rotation[:, parents[j]], rest[j] - rest[parents[j]])
        error = float(np.linalg.norm(fk - joints, axis=-1).max())
        if error > 2e-5:
            raise ValueError('Decoded body and model FK disagree; do not silently replace source joints')
        return joints, rotation, vertices, shaped, rest, parents, weights, faces, beta, {'decoded_fk_max_error_m': error}
    if spec.get('pose_convention', 'absolute_axis_angle') != 'absolute_axis_angle':
        raise ValueError('Expand PCA/mean-offset hand poses locally; absolute axis-angle required')
    if 'expression' in raw and np.any(raw['expression'] != 0):
        raise ValueError('Nonzero SMPL-X expression requires an explicit expression-aware decoder')
    pose = _finite(raw.get('poses', raw.get('local_axis_angle')), 'poses')
    pose = pose.reshape(len(pose), -1, 3)[take]
    if pose.shape[1:] != (count, 3):
        raise ValueError('Full family pose required; body/face/finger channels must not be silently dropped')
    trans = _finite(raw.get('trans', raw.get('transl')), 'trans')[take]
    if trans.shape != (len(take), 3):
        raise ValueError('SMPL transl must be T x 3, distinct from pelvis world position')
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(take), count, 3, 3)
    global_r = local.copy(); joints = np.empty((len(take), count, 3)); joints[:, 0] = rest[0] + trans
    for j in range(1, count):
        global_r[:, j] = global_r[:, parents[j]] @ local[:, j]
        joints[:, j] = joints[:, parents[j]] + np.einsum('tij,j->ti', global_r[:, parents[j]], rest[j] - rest[parents[j]])
    posedirs = _finite(model['posedirs'], 'posedirs')
    if posedirs.shape == ((count - 1) * 9, len(shaped) * 3):
        posedirs = posedirs.T.reshape(len(shaped), 3, -1)
    if posedirs.shape != (len(shaped), 3, (count - 1) * 9):
        raise ValueError('Model pose blendshape basis does not match full pose topology')
    vertices = []
    # Bound host memory without changing per-frame LBS or its timestamps.
    for begin in range(0, len(take), 8):
        sl = slice(begin, begin + 8)
        features = (local[sl, 1:] - np.eye(3)).reshape(len(local[sl]), -1)
        posed = shaped[None] + np.einsum('tp,vcp->tvc', features, posedirs)
        translation = joints[sl] - np.einsum('tjcd,jd->tjc', global_r[sl], rest)
        result = np.zeros_like(posed)
        for j in range(count):
            result += weights[None, :, j, None] * (np.einsum('tcd,tvd->tvc', global_r[sl, j], posed) + translation[:, j, None])
        vertices.append(result)
    result = (joints, global_r, np.concatenate(vertices), shaped, rest, parents, weights, faces, beta, {})
    if 'fitted_joints' in raw:
        difference = np.linalg.norm(joints[:, :22] - raw['fitted_joints'][take], axis=-1)
        result[-1]['saved_fitted_body_agreement_max_m'] = float(difference.max())
        if difference.max() > 2e-5:
            raise ValueError('Current licensed model differs from saved fitted source; verify model provenance')
    return result


def _palms(family, rest, physical_r, spec):
    if 'wrist_axes_local' in spec:
        axes = _rotation(spec['wrist_axes_local'], 'wrist_axes_local')
        if axes.shape != (2, 3, 3):
            raise ValueError('wrist_axes_local requires left/right distal/transverse/normal columns')
        label = 'Explicit model-local wrist semantic calibration'
    elif family in {'smplh', 'smplx'}:
        delta = 3 if family == 'smplx' else 0
        axes = []
        for wrist, index, middle, pinky, sign in [(20, 22, 25, 28, 1.), (21, 37, 40, 43, -1.)]:
            distal = _unit(rest[middle + delta] - rest[wrist]); across = rest[index + delta] - rest[pinky + delta]
            normal = _unit(sign * np.cross(distal, across))
            axes.append(np.stack([distal, np.cross(normal, distal), normal], -1))
        axes = np.asarray(axes)
        label = 'Original shaped wrist/MCP bind anatomy; no finger-action generation'
    else:
        raise ValueError('SMPL has no MCP topology; provide explicit wrist_axes_local calibration')
    return physical_r[:, [20, 21]] @ axes, axes, label


def prepare_supplied_source(spec, output_dir, *, model_path=None, observed_work=None, source_manifest=None):
    """Create the backend body22 bundle from a trusted local source descriptor.

    Model path is a trusted caller argument (never an uploaded field). External
    family is retained; generated native H uses the same decoded input adapter.
    Nothing is downloaded, and no supplied body is routed through EgoAllo.
    """
    route = source_route({'motion_source': spec})
    if route['route']=='supplied-robot':
        from .robot_motion_source import prepare_robot_source
        return prepare_robot_source(spec,output_dir,observed_work=observed_work,source_manifest=source_manifest)
    if spec.get('world_up') not in {'y', 'z'} or spec.get('units') != 'm':
        raise ValueError('Explicit source world_up=y/z and units=m required')
    if model_path is None:
        raise ValueError('Licensed body model must be configured locally at the decoding endpoint')
    source = Path(spec['path']).resolve(); model_path = Path(model_path).resolve()
    digest = sha256(source)
    if spec.get('sha256') and spec['sha256'] != digest:
        raise ValueError('Supplied source SHA256 changed')
    raw = _npz(source); _validate_family(raw, spec['family'])
    times, indices, take = _clock(raw, spec)
    decoded = _decode(raw, spec, take, model_path)
    joints, rotation, vertices, shaped, rest, parents, weights, faces, beta, check = decoded
    # Native SMPL model-local axes remain Y-up. Change world basis once.
    basis = np.eye(3) if spec['world_up'] == 'y' else C
    p_y = joints @ basis.T; r_y = basis @ rotation
    p_z = p_y @ C; physical_r = C.T @ r_y
    mesh = vertices @ basis.T @ C
    palm, palm_local, palm_label = _palms(spec['family'], rest, physical_r, spec)
    floor = float(spec['floor_z_m'])
    if not np.isfinite(floor):
        raise ValueError('Explicit finite fixed source floor required; no fitted ground replacement')
    sole = []; sole_ids = []
    for pair in ([7, 10], [8, 11]):
        region = weights[:, pair].sum(1) > .7
        if not region.any():
            raise ValueError('Model has no foot support skinning region')
        ids = np.flatnonzero(region & (shaped[:, 1] <= shaped[region, 1].min() + .012))
        if len(ids) < 3:
            raise ValueError('Insufficient actual source foot vertices')
        sole_ids.append(ids); sole.append(np.percentile(mesh[:, ids, 2], 5, axis=1))
    sole = np.stack(sole, 1)
    offset = rest - rest[np.maximum(parents, 0)]; offset[0] = 0
    fps = 1. / float(np.median(np.diff(times)))
    common = dict(timestamps_s=times, source_frame_indices=indices, exact_fps=np.array(fps), nominal_fps=np.array(30.))
    task_wrist = p_z[:, [20, 21]]
    task_forward, task_normal = palm[..., 0], palm[..., 2]
    observed = Path(observed_work) if observed_work is not None else None
    observation_hashes = {}
    if observed is not None:
        original_obs = _npz(observed / 'observations.npz')
        task_wrist = _finite(original_obs['wrist_position'], 'observed wrist_position')
        task_forward = _finite(original_obs['palm_forward'], 'observed palm_forward')
        task_normal = _finite(original_obs['palm_normal'], 'observed palm_normal')
        if any(x.shape != (len(times), 2, 3) for x in [task_wrist, task_forward, task_normal]):
            raise ValueError('Original task references and supplied body must have identical frame counts')
        if not np.allclose(original_obs['timestamps_s'], times, atol=1e-6, rtol=0):
            raise ValueError('Original observations and body timestamps disagree; no silent retiming')
        for name in ['observations.npz', 'source_hands.npz', 'operation-accuracy/source-targets.npz']:
            if (observed / name).is_file():
                observation_hashes[name] = sha256(observed / name)
    # Six-point task view preserves observed wrists; body22 remains immutable
    # reconstructed/fitted geometry for the learned prior and surface terms.
    six_positions = p_z[:, ROLES].copy()
    six_positions[:, 2:4] = task_wrist
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Source bundles are immutable; use a fresh versioned directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + '.tmp-', dir=output.parent))
    try:
        def save(name, **values):
            np.savez_compressed(temporary / name, **values)
        save('native_source.npz', **common, names=np.asarray(NAMES), parents=parents[:22], roles=ROLES,
             rest_offsets_cm=offset[:22] * 100, full_body_position_yup_m=p_y[:, :22], full_body_rotation_yup=r_y[:, :22],
             source_body_position_world_m=p_z[:, :22], source_joint_frame_world_rotation=physical_r[:, :22],
             source_body_rotation_world_transfer=C.T @ r_y[:, :22] @ C,
             six_position_zup_m=six_positions, reference_world_R_zup=C.T @ r_y[:, ROLES] @ C,
             wrist_position_world_m=task_wrist, palm_forward=task_forward, palm_normal=task_normal,
             palm_frame_world=palm, palm_local_smpl=palm_local, sole_low_z_m=sole, floor_z_m=np.array(floor), betas=beta)
        save('source_mesh.npz', **common, vertices_world_m=mesh.astype(np.float32), faces=faces.astype(np.int32),
             fps=np.array(30.), floor_world_m=np.array(floor), wrist_position_world_m=p_z[:, [20, 21]],
             wrist_rotation_world=physical_r[:, [20, 21]], pelvis_position_world_m=p_z[:, 0],
             left_sole_vertex_ids=sole_ids[0], right_sole_vertex_ids=sole_ids[1])
        save('observations.npz', **common, head_position=np.concatenate([p_z[:1, 15], p_z[:, 15]]),
             head_rotation=np.concatenate([physical_r[:1, 15], physical_r[:, 15]]),
             wrist_position=p_z[:, [20, 21]], palm_position=p_z[:, [20, 21]],
             palm_forward=palm[..., 0], palm_normal=palm[..., 2], floor_z=np.array(floor))
        (temporary / 'operation-accuracy').mkdir()
        save('operation-accuracy/source-targets.npz', points=p_z[:, [20, 21], None])
        objects_provenance = None
        if spec.get('objects_path'):
            objects_path = Path(spec['objects_path'])
            obj = _npz(objects_path)
            if not np.array_equal(obj['timestamps_s'], times):
                raise ValueError('Object trajectories must already use the same source window and clock')
            positions = _finite(obj['positions_world_m'], 'object positions')
            rotations = _rotation(obj['rotations_world'], 'object rotations')
            number = len(obj['object_ids'])
            if positions.shape != (len(times), number, 3) or rotations.shape != (len(times), number, 3, 3):
                raise ValueError('Object trajectories have invalid synchronized dimensions')
            for i in range(number):
                v = _finite(obj[f'object_{i}_vertices_local_m'], 'object vertices')
                f = np.asarray(obj[f'object_{i}_faces'])
                if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer) or f.min() < 0 or f.max() >= len(v):
                    raise ValueError('Invalid embedded object mesh')
            shutil.copyfile(objects_path, temporary / 'objects.npz')
            objects_provenance = dict(sha256=sha256(objects_path), objects=number, world='Z-up m; local mesh axes unchanged')
        if observed is not None:
            for name in observation_hashes:
                shutil.copyfile(observed / name, temporary / name)
            if (observed / 'wuji').exists():
                (temporary / 'wuji').symlink_to((observed / 'wuji').resolve(), target_is_directory=True)
        from greenwich_motion_sdk.motion import MotionClip
        provenance = dict(schema=SCHEMA, policy=POLICY, sourcekind='supplied_smpl_family',
            source_body=f"Supplied {spec['family']} full body; {spec.get('evidence', 'unspecified')} source",
            human_model=spec['family'], source_format=spec['format'], source_evidence=spec.get('evidence', 'unspecified'),
            source_path=str(source), source_sha256=digest, source_model_sha256=sha256(model_path),
            source_completion=False, neural_source_decode=False, body_encoder_joints=22,
            retained_source_pose_joints=FAMILY_JOINTS[spec['family']], generated_human_model=None,
            frames=len(times), fps=30., exact_fps=fps, source_frame_indices=indices.tolist(),
            neutral_mesh_height_m=float(np.ptp(shaped[:, 1])), body_bone_lengths_m=np.linalg.norm(offset[:22], axis=-1).tolist(),
            source_clock='Preserved original timestamps and integer-selected frame indices; no interpolation',
            source_pose_or_translation_modified=False, source_uniform_similarity_applied=False,
            source_world_up=spec['world_up'], source_units='m', canonical_world='Y-up cm', native_world='Z-up m',
            floor_world_m=floor, floor_policy=spec.get('floor_policy', 'Explicit source floor; not retargeted or clamped'),
            source_support='Actual shaped foot skinning region, lowest12mm bind, framewise world-Z fifth percentile',
            palm_convention=palm_label, fingers='Source pose retained where supplied; never treated as observed finger actions',
            end_effector=spec.get('end_effector', 'none'), original_bvh_independent_validation=False,
            model_arrays_exported=False, source_checks=check, objects=objects_provenance, task_reference_evidence=('original observations; unchanged' if observed is not None else 'supplied fitted/reconstructed body wrist; not separate observations'), observed_source_sha256=observation_hashes, source_spec={k:v for k,v in spec.items() if k != 'path'})
        original_manifest = dict(source_manifest or {})
        if observed is not None and (observed / 'input_manifest.json').is_file():
            original_manifest.update(json.loads((observed / 'input_manifest.json').read_text()))
        for key in ['clip', 'dataset', 'video', 'source_window', 'end_effector', 'declared_ground']:
            if key in original_manifest:
                provenance[key] = original_manifest[key]
        MotionClip(NAMES, parents[:22], offset[:22] * 100, r_y[:, :22], p_y[:, :22] * 100, times, provenance).save(temporary / 'canonical.npz')
        provenance['native_source_sha256'] = sha256(temporary / 'native_source.npz')
        provenance['source_mesh_sha256'] = sha256(temporary / 'source_mesh.npz')
        provenance['output_sha256'] = {str(p.relative_to(temporary)): sha256(p) for p in temporary.rglob('*.npz')}
        for name in ['source_provenance.json', 'provenance.json', 'input_manifest.json']:
            (temporary / name).write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
        if output.exists():
            output.rmdir()
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def prepare_body_source(work, output_dir, *, model_paths=None):
    """Normalize explicit supplied body or completed native-H work, never infer.

    Caller runs existing body_inference.run only when source_route returns the
    three-point route and no completed native H exists, then calls this helper.
    """
    work = Path(work)
    manifest = json.loads((work / 'input_manifest.json').read_text())
    if manifest.get('motion_source'):
        spec = dict(manifest['motion_source'])
        path = Path(spec['path'])
        if not path.is_absolute():
            from .dataset_io.storage import within
            path = within(work, path)
        spec['path'] = str(path)
        if spec.get('objects_path') and not Path(spec['objects_path']).is_absolute():
            from .dataset_io.storage import within
            spec['objects_path'] = str(within(work, spec['objects_path']))
        if spec.get('format')=='robot_joint_state':
            from .robot_motion_source import prepare_robot_source
            return prepare_robot_source(spec,output_dir,observed_work=work,source_manifest=manifest)
        model = (model_paths or {}).get(spec['family']) or os.environ.get('ARENA_' + spec['family'].upper() + '_MODEL')
    else:
        from .human_reference import family_for_work
        family_for_work(work)  # Reject retired H->X caches before adapting.
        human = work / 'smplh_motion.npz'
        if not human.is_file():
            raise FileNotFoundError('No supplied body; run the existing three-point native-H reconstruction first')
        observations = _npz(work / 'observations.npz')
        spec = dict(format='smpl_decoded', family='smplh', path=str(human), units='m', world_up='z',
                    evidence='reconstructed', floor_z_m=float(observations['floor_z']),
                    floor_policy=manifest.get('floor_policy', 'Declared source floor'), end_effector=manifest.get('end_effector', 'none'))
        model = work / 'model.npz'
    return prepare_supplied_source(spec, output_dir, model_path=model, observed_work=work if (work / 'observations.npz').is_file() else None, source_manifest=manifest)


def prepare_source(source_work, output_dir, *, model_paths=None):
    """Integration alias: source_work is an explicit private input workspace."""
    return prepare_body_source(source_work, output_dir, model_paths=model_paths)


def validate_three_point_observations(work):
    """Require actual head and two wrist observations before native-H inference."""
    d = _npz(Path(work) / 'observations.npz')
    wrist = _finite(d['wrist_position'], 'wrist_position')
    if wrist.ndim != 3 or wrist.shape[1:] != (2, 3):
        raise ValueError('Both wrist trajectories are required')
    n = len(wrist)
    for key, tail in [('head_position', (3,)), ('head_rotation', (3, 3))]:
        value = _finite(d[key], key)
        if value.shape not in [(n, *tail), (n + 1, *tail)]:
            raise ValueError('Head pose is required; hands alone must not invent a head')
    _rotation(d['head_rotation'], 'head_rotation')
    times = _finite(d['timestamps_s'], 'timestamps_s')
    if times.shape != (n,) or n < 2 or not np.allclose(np.diff(times), 1 / 30, atol=1e-6):
        raise ValueError('Three-point observations require synchronized 30 Hz timestamps')
    return {'frames': n, 'route': 'three-point-egoallo', 'generated_human_model': 'smplh'}


def describe_source(source, *, model_paths=None):
    """Read-only morphology metadata without LBS, neural inference or a robot.

    Accept an input workspace or a trusted supplied descriptor. Pose arrays may
    be read to establish timing; vertices are never animated by this function.
    """
    if not isinstance(source, dict):
        work = Path(source)
        native = work / 'native_source.npz'
        if native.is_file():
            with np.load(native, allow_pickle=False) as d:
                offsets = np.asarray(d['rest_offsets_cm'], float) / 100
                parents = np.asarray(d['parents'], int)
                rest = np.zeros_like(offsets)
                for j in range(1, len(rest)):
                    rest[j] = rest[parents[j]] + offsets[j]
                metadata_path = work / 'input_manifest.json'
                metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
                return dict(neutral_mesh_height_m=metadata.get('neutral_mesh_height_m'), source_sha256=sha256(native), body_bone_lengths_m=np.linalg.norm(offsets, axis=-1).tolist(),
                            neutral_joint_extent_m=float(np.ptp(rest[:, 1])), units='m', world_up='y',
                            height_kind='joint-centre extent, not physical mesh height', source_completion=False)
        manifest = json.loads((work / 'input_manifest.json').read_text())
        if not manifest.get('motion_source'):
            return dict(route='three-point-egoallo', morphology_available=False, generated_human_model='smplh')
        source = dict(manifest['motion_source'])
        if not Path(source['path']).is_absolute():
            from .dataset_io.storage import within
            source['path'] = str(within(work, source['path']))
    route = source_route({'motion_source': source})
    if route['route']=='supplied-robot':
        from .robot_motion_source import read_state, source_robot
        from .source_requirements import _robot_height
        row=source_robot(source);target,_,times,_=read_state(source,row)
        return dict(route='supplied-robot',family='robot',source_robot_id=source['robot_id'],
                    source_sha256=sha256(source['path']),source_model_sha256=target['model_sha256'],
                    frames=len(times),exact_fps=float(1/np.median(np.diff(times))),source_completion=False,
                    morphology_available=True,neutral_mesh_height_m=_robot_height(row),units='m',
                    body_bone_lengths_m=(np.linalg.norm(target['spec'].rest_offsets,axis=-1)/100).tolist(),
                    evidence=source.get('evidence','unspecified'))
    raw = _npz(source['path']); _validate_family(raw, source['family'])
    times, indices, take = _clock(raw, source)
    model_path = (model_paths or {}).get(source['family']) or os.environ.get('ARENA_' + source['family'].upper() + '_MODEL')
    result = dict(route='supplied-body', family=source['family'], source_sha256=sha256(source['path']),
                  frames=len(times), exact_fps=float(1 / np.median(np.diff(times))), source_completion=False,
                  morphology_available=model_path is not None, evidence=source.get('evidence', 'unspecified'))
    if model_path is not None:
        _, shaped, rest, parents, _, _ = _model(model_path, source['family'], _beta(raw))
        offsets = rest - rest[np.maximum(parents, 0)]; offsets[0] = 0
        result.update(neutral_mesh_height_m=float(np.ptp(shaped[:, 1])),
                      body_bone_lengths_m=np.linalg.norm(offsets[:22], axis=-1).tolist(),
                      source_model_sha256=sha256(model_path), units='m')
    return result
