"""Export decoded source geometry only; no body models, weights or synthetic RGB."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .settings import paths
from .motion_sources import sha256


def _asset(path):
    path = Path(path).resolve()
    try:
        relative = path.relative_to(paths.store.resolve())
    except ValueError as error:
        raise ValueError('Source preview destination must be inside the configured asset store') from error
    return '/assets/' + relative.as_posix()


def _immutable_bytes(path, data):
    path = Path(path)
    expected = hashlib.sha256(data).hexdigest()
    if path.exists():
        if sha256(path) != expected:
            raise ValueError('Refusing to replace different source preview geometry')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    return expected


def export_source_preview(normalized_source, destination):
    """Return source_body for scene.json; all exported positions stay world Z-up.

    No coordinate change, morphology scale, smoothing or pose fitting occurs.
    The viewer applies scene.source_transform once to original-space geometry.
    Source object GLBs preserve their local axes; their animation is separate.
    """
    source = Path(normalized_source)
    if (source / 'source_robot_playback.npz').is_file():
        return _export_robot_preview(source, Path(destination))
    mesh_path = source / 'source_mesh.npz'
    with np.load(mesh_path, allow_pickle=False) as data:
        vertices = np.asarray(data['vertices_world_m'], dtype='<f4')
        raw_faces = np.asarray(data['faces'])
        times = np.asarray(data['timestamps_s'], dtype=np.float64)
        indices = np.asarray(data['source_frame_indices'], dtype=np.int64)
        fps = float(data.get('exact_fps', data.get('fps', 30.)))
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Source vertices must be finite T x V x 3 world coordinates')
    n, v, _ = vertices.shape
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not np.issubdtype(raw_faces.dtype, np.integer) or raw_faces.min() < 0 or raw_faces.max() >= v:
        raise ValueError('Source triangles do not index the supplied vertices')
    if times.shape != (n,) or indices.shape != (n,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Source preview clock must be finite and frame-aligned')
    faces = np.asarray(raw_faces, dtype='<u4')
    provenance_path = source / 'source_provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    robot_source = provenance.get('source_format') == 'robot_joint_state'
    recorded_proxy = provenance.get('source_format') == 'recorded_skeleton_surface'
    digest = sha256(mesh_path)
    # Keep earlier immutable robot previews intact when adding source identity.
    target = Path(destination) / 'source-body' / (digest[:16] + ('-robot-v1' if robot_source else '-recorded-skeleton-v1' if recorded_proxy else ''))
    _asset(target)  # Check the boundary before writing anything.
    hashes = {
        'vertices': _immutable_bytes(target / 'vertices.bin', vertices.tobytes(order='C')),
        'faces': _immutable_bytes(target / 'faces.bin', faces.tobytes(order='C')),
        'clock': _immutable_bytes(target / 'clock.json', (json.dumps(dict(timestamps_s=times.tolist(),source_frame_indices=indices.tolist()))+'\n').encode())}
    result = dict(schema='arenalab.source-body-preview.v1', vertices_url=_asset(target/'vertices.bin'),
        faces_url=_asset(target/'faces.bin'), clock_url=_asset(target/'clock.json'),
        frames=n, vertex_count=v, face_count=len(faces), world_up='z', units='m',
        fps=30., exact_fps=fps, vertices_dtype='float32-le', faces_dtype='uint32-le',
        source_mesh_sha256=digest, output_sha256=hashes, source_policy=provenance.get('policy'),
        source_family='robot' if robot_source else provenance.get('human_model'), source_evidence=provenance.get('source_evidence'),
        source_sha256=provenance.get('source_sha256'),
        source_uniform_similarity_applied=bool(provenance.get('source_uniform_similarity_applied',False)),
        scope='Decoded source mesh only; no original RGB, body model or weights')
    if result['source_family'] == 'robot' and provenance.get('source_robot_id'):
        result['source_robot_id'] = provenance['source_robot_id']
    if recorded_proxy:
        result.update(source_format='recorded_skeleton_surface',
                      source_surface='skinning_joint_offsets_v1; approximate girth, not fitted subject surface',
                      scope='Recorded skeleton with approximate skinned surface; not original subject SMPL geometry')
    objects_path = source / 'objects.npz'
    if objects_path.is_file():
        result['objects'] = _export_objects(objects_path, target/'objects', times)
    _immutable_bytes(target/'manifest.json', (json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    return result


def _export_robot_preview(source, destination):
    """Publish exact rigid-link FK; never expand a robot into per-frame vertices."""
    from scipy.spatial.transform import Rotation
    provenance = json.loads((source/'source_provenance.json').read_text())
    model = source/'source_robot.glb'; motion = source/'source_robot_playback.npz'
    if sha256(model) != provenance['source_model_glb_sha256'] or sha256(motion) != provenance['source_visual_poses_sha256']:
        raise ValueError('Immutable robot preview changed')
    with np.load(motion, allow_pickle=False) as z:
        P, R, times = z['positions'], z['rotations'], z['timestamps_s']
    if (P.ndim != 3 or P.shape[-1] != 3 or R.shape != (*P.shape[:2],3,3)
            or len(times) != len(P) or not np.isfinite(P).all() or not np.isfinite(R).all()
            or not np.allclose(np.diff(times),1/30,atol=1e-8,rtol=0)):
        raise ValueError('Invalid source robot preview geometry/clock')
    role = provenance.get('source_video_role', provenance.get('motion_source', {}).get('video_role'))
    if role is not None and role not in {'recorded', 'preview-only', 'unspecified'}:
        raise ValueError('Unknown source video role')
    # Role is part of this immutable manifest's identity; prior previews remain.
    identity_fields = dict(motion=sha256(motion),model=sha256(model),
        objects=sha256(source/'objects.npz') if (source/'objects.npz').is_file() else None,
        video_role=role)
    source_metadata = {key:provenance[key] for key in ('source_evidence','source_stage')
                       if provenance.get(key) is not None}
    if any(not isinstance(value,str) or not value for value in source_metadata.values()):
        raise ValueError('Native source evidence and stage must be declared strings')
    # Preserve earlier paths/manifests when neither annotation was supplied.
    if source_metadata:identity_fields['source_metadata'] = source_metadata
    identity = hashlib.sha256(json.dumps(identity_fields,sort_keys=True).encode()).hexdigest()
    target = destination/'source-robot'/identity[:24]
    _asset(target)
    model_hash = _immutable_bytes(target/'model.glb',model.read_bytes())
    payload = dict(fps=30.,frames=len(times),positions=P.tolist(),
        quaternions=Rotation.from_matrix(R.reshape(-1,3,3)).as_quat().reshape(*P.shape[:2],4).tolist())
    playback_hash = _immutable_bytes(target/'playback.json',(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n').encode())
    result = dict(source_family='robot',source_robot_id=provenance['source_robot_id'],world_up='z',
        frames=len(times),fps=30.,model_url=_asset(target/'model.glb'),playback_url=_asset(target/'playback.json'),
        model_sha256=model_hash,playback_sha256=playback_hash,source_sha256=provenance['source_sha256'],
        retargeted=False,source_trajectory_modified=False)
    if role is not None:result['video_role']=role
    if source_metadata:
        result.update(source_metadata, preview_retargeting_performed=False)
    if (source/'objects.npz').is_file():
        result['objects'] = _export_objects(source/'objects.npz',target/'objects',times)
    _immutable_bytes(target/'manifest.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    return result


def _export_objects(path, destination, times):
    """Small portable GLBs plus original object-local-to-world trajectories."""
    import trimesh
    from scipy.spatial.transform import Rotation
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data['timestamps_s'], times):
            raise ValueError('Source body and objects must share the exact clock')
        ids = [str(value) for value in data['object_ids']]
        positions = np.asarray(data['positions_world_m'],float)
        rotations = np.asarray(data['rotations_world'],float)
        if positions.shape!=(len(times),len(ids),3) or rotations.shape!=(len(times),len(ids),3,3) or not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError('Invalid source object trajectories')
        result=[]
        for i,name in enumerate(ids):
            vertices=np.asarray(data[f'object_{i}_vertices_local_m'],float)
            faces=np.asarray(data[f'object_{i}_faces'],np.int64)
            if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(vertices):
                raise ValueError('Invalid embedded object mesh')
            folder=Path(destination)/str(i)
            glb=trimesh.Trimesh(vertices=vertices,faces=faces,process=False).export(file_type='glb')
            glb_sha=_immutable_bytes(folder/'mesh.glb',glb)
            animation=dict(timestamps_s=times.tolist(),position=positions[:,i].tolist(),
                quaternion_xyzw=Rotation.from_matrix(rotations[:,i]).as_quat().tolist(),world_up='z',units='m')
            pose_sha=_immutable_bytes(folder/'poses.json',(json.dumps(animation)+'\n').encode())
            result.append(dict(id=name,mesh_url=_asset(folder/'mesh.glb'),poses_url=_asset(folder/'poses.json'),
                frames=len(times),world_up='z',units='m',mesh_sha256=glb_sha,poses_sha256=pose_sha,
                local_axes='original object mesh; no rotation or scaling baked into vertices'))
        return result
