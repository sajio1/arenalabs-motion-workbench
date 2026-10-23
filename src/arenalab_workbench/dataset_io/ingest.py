"""Validated uploads. Original archives remain immutable outside the served directory."""
from arenalab_workbench.settings import paths
from pathlib import Path
import hashlib
import json
import stat
import zipfile
import numpy as np
from .storage import within, write

MAX_ARCHIVE = 2 * 1024**3
MAX_EXPANDED = 8 * 1024**3
MAX_FRAMES = 108000  # One hour at the current 30 Hz processing rate.


def unpack(archive, destination):
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) > 20000 or sum(m.file_size for m in members) > MAX_EXPANDED:
            raise ValueError('Package exceeds the expanded size or file-count limit')
        paths = set()
        for item in members:
            if stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 1:
                raise ValueError('Encrypted files and symlinks are not supported')
            target = within(destination, item.filename)
            if target in paths:
                raise ValueError('Duplicate paths in upload')
            paths.add(target)
        for item in members:
            target = within(destination, item.filename)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(item) as source, target.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(source, output, 1024 * 1024)


def array(data, key, shape):
    value = np.asarray(data[key], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{key} must contain finite values with shape {shape}')
    return value


def rotations(value, name):
    if not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(value), 1, atol=1e-4):
        raise ValueError(f'{name} must contain right-handed rotation matrices')


def _validated_video(package, manifest, n):
    video = within(package, manifest['video'])
    if not video.is_file() or video.suffix.lower() != '.mp4':
        raise ValueError('Include the synchronized original MP4 video')
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    start = float(manifest.get('video_start_seconds', 0))
    if not np.isfinite(start) or start < 0 or fps <= 0 or count / fps + 1 / fps < start + n / 30:
        raise ValueError('Video duration does not cover all trajectory timestamps')
    cap=cv2.VideoCapture(str(video))
    try:
        for seconds in [start,start+(n-1)/30]:
            cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
            ok,image=cap.read()
            if not ok or image is None:raise ValueError('Video cannot decode the required trajectory interval')
    finally:cap.release()
    if manifest.get('image_size') != [width, height]:
        raise ValueError('image_size must match the original video [width,height]')
    return video


def validate_normalized(package):
    manifest = json.loads((Path(package) / 'recording.json').read_text())
    if manifest.get('schema') != 'arenalabs.recording.v1':
        raise ValueError('Unsupported recording schema; download the upload template')
    if manifest.get('fps') != 30 or manifest.get('units') != 'm' or manifest.get('world_up') != 'z':
        raise ValueError('Declare fps=30, units=m, world_up=z; convert coordinates before upload')
    mode = manifest.get('end_effector')
    if mode not in {'none', 'wuji', 'robotiq-2f-140'}:
        raise ValueError('Declare none, wuji, or robotiq-2f-140 as end_effector')
    from arenalab_workbench.motion_sources import validate_body_attachment, _npz, _clock
    supplied = validate_body_attachment(package, manifest)
    if supplied is not None:
        # Registry-only absolute descriptor, resolved after the package boundary
        # check. The user's model is never loaded from an upload-selected path.
        manifest['_resolved_motion_source'] = supplied
    observation_path = Path(package) / manifest.get('observations', 'observations.npz')
    if supplied is not None and not observation_path.is_file():
        if mode != 'none':
            raise ValueError('Finger/gripper components require their own observed inputs')
        times, _, _ = _clock(_npz(supplied['path']), supplied)
        manifest['frames'] = len(times)
        video = _validated_video(package, manifest, len(times)) if manifest.get('video') else None
        return manifest, dict(timestamps_s=times, floor_z=np.array(float(supplied['floor_z_m']))), video
    path = within(package, manifest.get('observations', 'observations.npz'))
    # Check NPZ headers before NumPy allocation, disallow pickled object arrays.
    with zipfile.ZipFile(path) as zipped:
        if sum(m.file_size for m in zipped.infolist()) > 512 * 1024**2:
            raise ValueError('Observation tensors exceed 512 MiB')
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data['timestamps_s'], float)
        n = len(times)
        if not 2 <= n <= MAX_FRAMES or times.shape != (n,) or not np.isfinite(times).all() or not np.allclose(times - times[0], np.arange(n) / 30, atol=1e-4):
            raise ValueError('timestamps_s must be a finite, continuous 30 Hz clock')
        p = array(data, 'wrist_position', (n, 2, 3))
        camera = array(data, 'camera_to_world', (n, 4, 4))
        rotations(camera[:, :3, :3], 'camera_to_world')
        if not np.allclose(camera[:, 3], [0, 0, 0, 1]):
            raise ValueError('Invalid homogeneous camera poses')
        k = array(data, 'intrinsic', (3, 3))
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError('Invalid pinhole camera intrinsic matrix')
        orientation = array(data, 'ee_rotation', (n, 2, 3, 3))
        rotations(orientation, 'ee_rotation')
        floor = float(manifest['floor_z_m'])
        if not np.isfinite(floor):
            raise ValueError('Declare a finite floor_z_m, using trajectory preview if needed')
        if supplied is not None:
            body_times, _, _ = _clock(_npz(supplied['path']), supplied)
            if body_times.shape != times.shape or not np.allclose(body_times, times, atol=1e-6, rtol=0):
                raise ValueError('Supplied body and observed input clocks must match exactly')
        result = dict(timestamps_s=times if supplied is not None else times-times[0], camera=camera, intrinsic=k, wrist_position=p, ee_rotation=orientation, floor_z=np.array(floor))
        if mode == 'robotiq-2f-140':
            width = array(data, 'gripper_width_m', (n, 2))
            if np.any(width < 0) or np.any(width > .3):
                raise ValueError('Invalid gripper widths in metres')
            result['width_m'] = width
        if mode == 'wuji':
            result['hand_points'] = array(data, 'hand_points', (n, 2, 21, 3))
            result['hand_confidence'] = array(data, 'hand_confidence', (n, 2, 21)) if 'hand_confidence' in data else np.ones((n, 2, 21))
    video = _validated_video(package, manifest, n)
    return manifest, result, video


def prepare_body(record, work):
    """Use the same EgoAllo input and observed palm/finger refinement as the demo."""
    from arenalab_workbench.settings import paths
    BASE = paths.egoallo
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if record.get('body_adapter') == 'head-wrists-v1':
        # Server-registered measured poses need no fabricated optical camera.
        # These are observations for native-H completion, not a supplied body.
        from arenalab_workbench.motion_sources import validate_three_point_observations
        import shutil
        source = Path(record['source_row']['source'])
        if record['public']['end_effector'] != 'none' or record.get('motion_source'):
            raise ValueError('Explicit head/wrist adapter requires observed poses and no hand')
        with np.load(source, allow_pickle=False) as saved:
            data = {name: saved[name] for name in saved.files}
        n = len(data['wrist_position'])
        if data['head_position'].shape != (n+1,3) or data['head_rotation'].shape != (n+1,3,3):
            raise ValueError('Explicit head/wrist source requires the native-H leading head sample')
        for name in ('palm_position','palm_forward','palm_normal'):
            value = np.asarray(data[name])
            if value.shape != (n,2,3) or not np.isfinite(value).all():
                raise ValueError('Missing or invalid observed palm geometry: '+name)
        if np.any(np.linalg.norm(np.cross(data['palm_forward'],data['palm_normal']),axis=-1)<1e-6):
            raise ValueError('Degenerate observed palm directions')
        if not np.isfinite(float(data['floor_z'])):
            raise ValueError('Explicit source floor required')
        shutil.copyfile(source,work/'observations.npz')
        validate_three_point_observations(work)
        (work/'operation-accuracy').mkdir(exist_ok=True)
        np.savez_compressed(work/'operation-accuracy/source-targets.npz',points=data['wrist_position'][:,:,None])
        provenance=record.get('manifest',{})
        write(work/'input_manifest.json',dict(dataset=record['public'].get('dataset','Uploaded'),clip=record['public']['id'],frames=n,fps=30.,end_effector='none',human_model='smplh',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_trajectory_edited=False,video=record['source_row'].get('video'),body_inputs=['observed head pose','observed left wrist pose','observed right wrist pose'],body_adapter='head-wrists-v1',source_provenance=provenance,cpf_policy=provenance.get('head_reference','Declared source head frame; no invented camera'),floor_policy=provenance.get('floor_policy','Declared source floor'),excluded=['recorded pelvis','recorded feet','recorded body joint poses','reconstructed source SMPL']))
        for name in ['model.npz','egoallo_checkpoint_april13']:
            if not (work/name).exists():(work/name).symlink_to(BASE/name,target_is_directory=(BASE/name).is_dir())
        return work
    supplied = record.get('motion_source') or record.get('manifest', {}).get('_resolved_motion_source')
    if supplied is not None:
        write(work / 'input_manifest.json', dict(dataset=record['public'].get('dataset', 'Uploaded'), clip=record['public']['id'],
            frames=record['source_row']['frames'], fps=30., end_effector=record['public']['end_effector'],
            motion_source=supplied, source_sha256=supplied['sha256'], source_trajectory_edited=False,
            video=record['source_row'].get('video'), floor_policy='Explicit uploaded source floor'))
    source = np.load(record['source_row'].get('observations_source', record['source_row']['source']), allow_pickle=False)
    if supplied is not None and 'wrist_position' not in source:
        return work  # Fully supplied body, no fabricated observation/reconstruction.
    p = source['wrist_position']
    mode = record['public']['end_effector']
    camera = source['camera']
    head_r = camera[:, :3, :3] @ np.diag([-1., -1., 1.])
    forward = source['ee_rotation'][..., 2]
    normal = source['ee_rotation'][..., 0]
    palm = p
    if mode == 'wuji':
        points = source['hand_points']
        palm = points[:, :, [5, 9, 13, 17]].mean(2)
        np.savez_compressed(work / 'source_hands.npz', points=points, confidence=source['hand_confidence'], source_to_zup=np.eye(3))
        tips = points[:, :, [4, 8, 12, 16, 20]]
        targets = np.concatenate([palm[:, :, None], tips, tips[:, :, :2].mean(2)[:, :, None]], 2)
    else:
        targets = p[:, :, None]
    np.savez_compressed(work / 'observations.npz', head_position=np.r_[camera[:1, :3, 3], camera[:, :3, 3]], head_rotation=np.concatenate([head_r[:1], head_r]), wrist_position=p, palm_position=palm, palm_forward=forward, palm_normal=normal, timestamps_s=source['timestamps_s'], floor_z=source['floor_z'])
    (work / 'operation-accuracy').mkdir(exist_ok=True)
    np.savez_compressed(work / 'operation-accuracy/source-targets.npz', points=targets)
    write(work / 'input_manifest.json', dict(dataset=record['public'].get('dataset', 'Uploaded'), clip=record['public']['id'], frames=len(p), fps=30, end_effector=mode, source_sha256=hashlib.sha256(Path(record['source_row']['source']).read_bytes()).hexdigest(), source_trajectory_edited=False, source_window=None, video=record['source_row']['video'], cpf_policy='User-declared optical head camera used as head proxy', floor_policy='User-declared floor'))
    if supplied is not None:
        manifest = json.loads((work / 'input_manifest.json').read_text())
        manifest['motion_source'] = supplied
        write(work / 'input_manifest.json', manifest)
        if mode == 'wuji' and not (work / 'wuji').exists():
            (work / 'wuji').symlink_to(BASE / 'wuji', target_is_directory=True)
        return work
    for name in ['model.npz', 'egoallo_checkpoint_april13', 'wuji']:
        if not (work / name).exists():
            (work / name).symlink_to(BASE / name, target_is_directory=(BASE / name).is_dir())
