"""Validate a complete downloaded reference dataset without models or cloud access.

Ready means consistent, complete reference data, including separately mounted
component channels. It does not mean a controller or hand physics was executed.
Requires NumPy and PyArrow. The exported copy locates its native reader locally.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import runpy
from urllib.parse import urlsplit

import numpy as np

MODALITIES = ('trajectory', 'point_cloud', 'contact', 'inpaint_fov')
SCHEMA = 'arenalabs.dataset-readiness.v1'


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('Duplicate JSON key in dataset')
        value[key] = item
    return value


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs)


def _path(root, relative, *, directory=False):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Missing package-relative artifact path')
    parsed = urlsplit(relative)
    path = PurePosixPath(parsed.path)
    if (parsed.scheme or parsed.netloc or parsed.fragment or parsed.query or
            '\\' in relative or ':' in relative or path.is_absolute() or
            '..' in path.parts or str(path) != relative):
        raise ValueError('Unsafe or external package artifact path')
    target = root.joinpath(*path.parts)
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Package artifact escapes dataset')
    if not (target.is_dir() if directory else target.is_file()):
        raise ValueError('Referenced package artifact is missing')
    return target


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _checksums(root):
    listed = _read(root / 'checksums.json')
    if not isinstance(listed, dict) or not listed or 'checksums.json' in listed:
        raise ValueError('Invalid dataset checksum manifest')
    actual = set()
    for entry in root.rglob('*'):
        if entry.is_symlink():
            raise ValueError('Dataset must not contain symlinks')
        if entry.is_file() and entry != root / 'checksums.json':
            actual.add(entry.relative_to(root).as_posix())
    for name, digest in listed.items():
        path = _path(root, name)
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('Invalid artifact SHA256')
        if _sha(path) != digest:
            raise ValueError('Dataset artifact checksum mismatch: ' + name)
    if set(listed) != actual:
        raise ValueError('Dataset checksum inventory has missing or extra files')
    return listed


def _clock(value, frames, label):
    if value.get('frames') != frames or value.get('fps') != 30:
        raise ValueError(label + ' clock differs from the 30 Hz dataset')


def _scene_references(root, value):
    """Verify portable viewer dependencies without following provenance paths."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and (key == 'url' or key.endswith('_url') or key == 'native_original'):
                _path(root, item, directory=key == 'geometry_url')
            else:
                _scene_references(root, item)
    elif isinstance(value, list):
        for item in value:
            _scene_references(root, item)


def _modality_contract(root, scene, manifest, robot_id):
    selected = scene.get('output_modalities')
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or not set(selected).issubset(MODALITIES) or 'trajectory' not in selected):
        raise ValueError('Export requires an explicit selected trajectory and output set')
    if set(scene.get('completed_outputs', [])) != set(selected):
        raise ValueError('Selected outputs are incomplete')
    statuses = scene.get('output_status', {})
    if any(statuses.get(name) != 'completed' for name in selected):
        raise ValueError('Selected output status is not completed')
    if scene.get('generation_status', {}).get('status', 'completed') != 'completed':
        raise ValueError('Scene generation is incomplete')
    refs = {'trajectory': [manifest['native_reference']['contract']]}
    if 'point_cloud' in selected:
        cloud = scene.get('cloud') or {}
        ref = cloud.get('geometry_url') or cloud.get('url')
        _path(root, ref, directory=bool(cloud.get('geometry_url')))
        refs['point_cloud'] = [ref]
    if 'contact' in selected:
        ref = (scene.get('tactile') or {}).get('url')
        _path(root, ref)
        refs['contact'] = [ref]
    if 'inpaint_fov' in selected:
        ref = scene.get('patch_videos', {}).get(robot_id)
        if (scene.get('patch_unavailable', {}).get(robot_id) or
                scene.get('patch_kinds', {}).get(robot_id) == 'native_preview'):
            raise ValueError('Requested inpaint FOV is unavailable')
        _path(root, ref)
        refs['inpaint_fov'] = [ref]
    result = {name: {'status': 'ready' if name in selected else 'not_selected',
                     'paths': refs.get(name, [])} for name in MODALITIES}
    declared = manifest.get('modalities')
    if declared is not None:
        if set(declared) != set(MODALITIES):
            raise ValueError('Modality manifest must contain all four output statuses')
        for name, expected in result.items():
            entry = declared[name]
            if entry.get('status') != expected['status']:
                raise ValueError('Modality manifest disagrees with selected output status')
            paths = entry.get('paths', [])
            if not isinstance(paths, list) or set(paths) != set(expected['paths']):
                raise ValueError('Modality manifest artifact paths differ')
    return result


def validate_dataset(root):
    """Return a ready receipt, or raise on incomplete/corrupt/ambiguous exports.

    Checks every listed file, every body and attachment joint sample, replay
    clocks, selected deliverables and the retained all-frame native FK receipt.
    No simulator, robot description, model inference or network is required.
    """
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    hashes = _checksums(root)
    manifest = _read(_path(root, 'extensions/arenalabs/manifest.json'))
    scene = _read(_path(root, 'extensions/arenalabs/scene.json'))
    frames = manifest.get('frames')
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError('Dataset requires a positive frame count')
    _clock(manifest, frames, 'Manifest')
    _clock(scene, frames, 'Scene')
    readiness = manifest.get('readiness', {})
    if (readiness.get('status') != 'ready' or readiness.get('scope') != 'complete_kinematic_reference_data'
            or readiness.get('dynamics_validated') is not False):
        raise ValueError('Dataset readiness declaration is missing or inconsistent')
    _path(root, readiness.get('validator'))
    if not isinstance(manifest.get('modalities'), dict):
        raise ValueError('Unified modality manifest is missing')
    if scene.get('id') != manifest.get('scene_id'):
        raise ValueError('Scene identity differs from exported recording')
    robots = scene.get('robots', [])
    if len(robots) != 1 or robots[0].get('id') != manifest.get('robot_id'):
        raise ValueError('Dataset must contain exactly the selected robot')
    robot = robots[0]
    reference = manifest.get('native_reference', {})
    if reference.get('available') is not True:
        raise ValueError('EXPORT_NATIVE_REFERENCE_UNAVAILABLE: visual-only data is not a ready reference export')
    if reference.get('physics_control_validated') is not False:
        raise ValueError('Reference export must not claim controller validation')
    native_path = _path(root, reference.get('contract'))
    reader = _path(root, (native_path.parent / 'native_reference_mujoco.py').relative_to(root).as_posix())
    _path(root, (native_path.parent / 'native_reference_isaac.py').relative_to(root).as_posix())
    # run_path does not create __pycache__ inside an immutable downloaded bundle.
    native, arrays = runpy.run_path(str(reader), run_name='arena_dataset_native_reader')['load'](native_path.parent)
    _clock(native, frames, 'Native reference')
    if native.get('robot_id') != robot['id'] or reference.get('frames') != frames:
        raise ValueError('Native reference identity/frame count differs')
    raw_files=manifest.get('raw_files')
    native_raw='extensions/arenalabs/raw/robot/refined/motion.npz'
    if (not isinstance(raw_files,list) or any(not isinstance(name,str) for name in raw_files) or
            len(set(raw_files))!=len(raw_files) or native_raw not in raw_files):
        raise ValueError('Authoritative native raw motion is not declared in the dataset')
    for name in raw_files:
        _path(root,name)
    if (native.get('source',{}).get('immutable_raw_retained') is not True or
            _sha(_path(root,native_raw))!=native.get('source',{}).get('sha256')):
        raise ValueError('Raw native motion SHA256 differs from the standardized reference source')
    physics = native.get('physics', {})
    if (physics.get('dynamic_tracking_validated') is not False or
            physics.get('physical_feasibility_validated') is not False):
        raise ValueError('Native reference must retain unvalidated physical execution status')
    receipt = _read(_path(root, (native_path.parent / 'validation.json').relative_to(root).as_posix()))
    if (receipt.get('kinematic_replay_passed') is not True or receipt.get('frames') != frames or
            receipt.get('checked_link_poses') != frames * len(native['link_names'])):
        raise ValueError('Native replay validation does not cover every frame and link')
    if receipt.get('dynamics_rollout_validated') is not False or receipt.get('controller_used') is not False:
        raise ValueError('Native kinematic receipt has inconsistent physics semantics')
    for key in ('position_max_m', 'quaternion_chord_max'):
        error = receipt.get(key)
        if not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 2e-6:
            raise ValueError('Native replay validation residual is invalid')
    playback = _read(_path(root, robot.get('playback_url')))
    _clock(playback, frames, 'Playback')
    poses=np.asarray(playback.get('positions'),dtype=float)
    quaternions=np.asarray(playback.get('quaternions'),dtype=float)
    if (poses.ndim!=3 or poses.shape[0]!=frames or poses.shape[1]<1 or poses.shape[2]!=3 or
            quaternions.shape!=poses.shape[:2]+(4,) or not np.isfinite(poses).all() or
            not np.isfinite(quaternions).all() or
            not np.allclose(np.linalg.norm(quaternions,axis=-1),1.,atol=1e-4,rtol=0)):
        raise ValueError('Visual replay mesh poses are incomplete or invalid')
    replay=manifest.get('replay',{})
    _path(root,replay.get('stage'))
    visual_contract=_read(_path(root,replay.get('contract')))
    _clock(visual_contract,frames,'Visual replay')
    if (visual_contract.get('meshes')!=poses.shape[1] or visual_contract.get('units')!='m' or
            visual_contract.get('world_up')!='Z' or visual_contract.get('handedness')!='right' or
            visual_contract.get('physics_articulation') is not False):
        raise ValueError('Visual replay contract differs from mesh pose data')
    names = playback['joints']['names']
    values = np.asarray(playback['joints']['values'], dtype=float)
    if (not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names) or values.shape != (frames, len(names)) or not np.isfinite(values).all()):
        raise ValueError('Invalid named playback state')
    extras = native['additional_joints']['names']
    if set(names) != set(native['joint_names']) | set(extras):
        raise ValueError('Body and attachment channels do not cover all playback joints')
    for keys in (('joint_names', 'joint_position'), ('additional_joint_names', 'additional_joint_position')):
        for index, name in enumerate(arrays[keys[0]].tolist()):
            if not np.allclose(values[:, names.index(name)], arrays[keys[1]][:, index], atol=1e-8, rtol=0):
                raise ValueError('Playback/native named joint position differs: ' + name)
    units = playback['joints'].get('units', playback['joints'].get('unit'))
    if isinstance(units, str):
        units = [units] * len(names)
    native_units = {joint['name']: joint['unit'] for joint in native['joints']}
    native_units.update(zip(extras, native['additional_joints']['units']))
    if units != [native_units[name] for name in names]:
        raise ValueError('Playback/native joint units differ')
    state_units = manifest.get('state_units')
    if isinstance(state_units, str):
        state_units = [state_units] * len(names)
    if state_units != units:
        raise ValueError('Dataset state units differ from named joint units')
    visual_receipt = _read(_path(root, reference.get('visual_validation')))
    count = visual_receipt.get('body_meshes')
    basis = np.asarray(scene.get('source_world_to_display'), dtype=float)
    if (basis.shape != (3, 3) or not np.isfinite(basis).all() or
            not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8, rtol=0) or
            not np.isclose(np.linalg.det(basis), 1., atol=1e-8, rtol=0) or
            not np.allclose(basis[:, 2], [0, 1, 0], atol=1e-8, rtol=0)):
        raise ValueError('Native visual scene basis is absent or invalid')
    if (visual_receipt.get('schema') != 'arenalabs-native-visual-validation-v1' or
            visual_receipt.get('passed') is not True or visual_receipt.get('frames') != frames or
            visual_receipt.get('robot_id') != robot['id'] or
            not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= poses.shape[1] or
            visual_receipt.get('total_meshes') != poses.shape[1] or
            visual_receipt.get('checked_body_mesh_poses') != frames * count or
            visual_receipt.get('source_sha256') != native['source']['sha256'] or
            visual_receipt.get('model_source_sha256') != native['model']['source_file_sha256'] or
            visual_receipt.get('playback_sha256') != _sha(_path(root, robot['playback_url'])) or
            visual_receipt.get('source_world_to_display') != basis.tolist() or
            visual_receipt.get('dynamics_validated') is not False or
            visual_receipt.get('attachment_fresh_fk_validated') is not False):
        raise ValueError('Native visual receipt is incomplete or differs from current reference/playback')
    visual_errors = ['body_position_max_m', 'body_rotation_matrix_max']
    if visual_receipt.get('saved_render_all_meshes_checked') is True:
        visual_errors += ['saved_render_position_max_m', 'saved_render_rotation_matrix_max']
    elif visual_receipt.get('saved_render_all_meshes_checked') is not False:
        raise ValueError('Native visual saved-render coverage is ambiguous')
    for key in visual_errors:
        error = visual_receipt.get(key)
        if not isinstance(error, (int, float)) or not np.isfinite(error) or not 0 <= error <= 2e-6:
            raise ValueError('Native visual validation residual is invalid')
    info = _read(_path(root, 'meta/info.json'))
    if info.get('fps') != 30 or info.get('total_frames') != frames or info.get('total_episodes') != 1 or info.get('robot_type') != robot['id']:
        raise ValueError('LeRobot metadata differs from exported reference')
    tables = sorted((root / 'data').rglob('*.parquet'))
    if not tables:
        raise ValueError('LeRobot frame table is missing')
    table = __import__('pyarrow').concat_tables([pq.read_table(path) for path in tables])
    data = table.to_pydict()
    for field in ('observation.state', 'action'):
        if info['features'][field].get('names') != names:
            raise ValueError('LeRobot named joint order differs from playback')
    state = np.asarray(data['observation.state'], dtype=np.float32)
    action = np.asarray(data['action'], dtype=np.float32)
    expected = values.astype(np.float32)
    if not np.array_equal(state, expected):
        raise ValueError('LeRobot current-frame state differs from playback')
    if not np.array_equal(action, np.concatenate([expected[1:], expected[-1:]])):
        raise ValueError('LeRobot action is not the next-frame reference with repeated terminal frame')
    for field in ('frame_index', 'index'):
        if not np.array_equal(data[field], np.arange(frames)):
            raise ValueError('LeRobot frame indices are incomplete or reordered')
    if (not np.array_equal(data['episode_index'], np.zeros(frames, dtype=int)) or
            not np.array_equal(np.asarray(data['timestamp'], dtype=np.float32), np.arange(frames, dtype=np.float32) / 30)):
        raise ValueError('LeRobot episode/timestamps differ from the 30 Hz clock')
    frame_map = pq.read_table(_path(root, manifest['frame_map'])).to_pydict()
    if (not np.array_equal(frame_map['frame_index'], np.arange(frames)) or
            not np.allclose(frame_map['timestamp'], np.arange(frames) / 30, atol=1e-10, rtol=0)):
        raise ValueError('Modality frame map does not cover the reference clock')
    cloud_frames=scene.get('cloud',{}).get('frames',[])
    source_ids=[entry.get('source_frame') for entry in cloud_frames]
    if (any(not isinstance(index,int) or isinstance(index,bool) or index<0 or index>=frames for index in source_ids)
            or any(b<=a for a,b in zip(source_ids,source_ids[1:]))):
        raise ValueError('Cloud observations have invalid or ambiguous source frame indices')
    if 'point_cloud' in scene.get('output_modalities',[]) and not source_ids:
        raise ValueError('Selected point cloud has no observed frames')
    nearest=np.searchsorted(source_ids,np.arange(frames),side='right')-1
    expected_cloud=[int(index) if index>=0 else None for index in nearest]
    expected_source=[source_ids[index] if index>=0 else None for index in nearest]
    if (frame_map.get('cloud_index')!=expected_cloud or frame_map.get('cloud_source_frame')!=expected_source or
            frame_map.get('cloud_observed_this_frame')!=[index==source for index,source in enumerate(expected_source)]):
        raise ValueError('Modality frame map disagrees with observed cloud frames')
    _scene_references(root, scene)
    modalities = _modality_contract(root, scene, manifest, robot['id'])
    return dict(schema=SCHEMA, ready=True, scope='complete_kinematic_reference_data',
                robot_id=robot['id'], recording_id=scene['id'], frames=frames, fps=30,
                verified_files=len(hashes), all_frames_checked=True,
                named_joint_count=len(names), body_joint_count=len(native['joint_names']),
                attachment_components=dict(names=extras, channels_retained=True,
                    applied_by_body_adapter=False, physics_validated=False,
                    default_replay_mode='body', required_only_for_hand_replay=bool(extras)),
                visual_reference=dict(body_meshes=count, all_body_frames_checked=True,
                    saved_render_all_meshes_checked=visual_receipt['saved_render_all_meshes_checked'],
                    attachment_fresh_fk_validated=False),
                modalities=modalities, dynamics_validated=False, controller_validated=False,
                verification='Offline integrity/schema/cross-representation checks plus retained all-frame native FK receipt; not a new simulator execution')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_dataset(args.dataset), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(schema=SCHEMA, ready=False, error=str(error))))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
