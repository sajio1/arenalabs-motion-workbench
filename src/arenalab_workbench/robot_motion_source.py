"""Complete native robot motion -> the existing AlphaMotion/refinement contract.

Only named joint states and a locally registered source robot are accepted.
The source model remains at the decoding endpoint. No human model, completion,
resampling, limit projection or target-specific policy is applied here.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np

FORMAT = 'robot_joint_state'
POLICY = 'supplied-robot-fullbody-v1'
C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def is_robot(spec):
    return isinstance(spec, dict) and spec.get('format') == FORMAT


def source_robot(spec):
    from .robot_registry import roster
    from arenalab_motion.robot_metadata import ensure_robot_metadata
    identifier = spec.get('robot_id')
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('Complete robot motion requires a locally registered source robot_id')
    row = roster().get(identifier)
    if row is None:
        raise ValueError('Source robot is unavailable at this decoding endpoint: ' + identifier)
    return ensure_robot_metadata(row['pipeline_row'])


def read_state(spec, row=None):
    from .motion_sources import _npz, _clock, sha256
    from arenalab_motion.prior import load_target
    if not is_robot(spec) or spec.get('world_up') != 'z' or spec.get('units') != 'm':
        raise ValueError('Robot joint state requires format=robot_joint_state, world_up=z and units=m')
    floor = float(spec.get('floor_z_m', float('nan')))
    if not np.isfinite(floor):
        raise ValueError('Robot motion requires a finite declared floor_z_m')
    path = Path(spec['path'])
    if spec.get('sha256') and sha256(path) != spec['sha256']:
        raise ValueError('Original robot motion hash changed')
    raw = _npz(path)
    required = {'joint_positions', 'joint_names', 'root_position', 'root_quaternion_wxyz', 'timestamps_s'}
    if required - raw.keys():
        raise ValueError('Complete robot motion is missing: ' + ', '.join(sorted(required - raw.keys())))
    times, indices, take = _clock(raw, spec)
    if len(times) < 3:
        raise ValueError('Robot motion requires at least three 30 Hz frames')
    names = raw['joint_names']
    if names.ndim != 1 or names.dtype.kind not in 'US' or len(set(names.tolist())) != len(names):
        raise ValueError('Source joint_names must be unique strings')
    n = len(raw['timestamps_s'])
    for key, shape in [('joint_positions', (n, len(names))), ('root_position', (n, 3)), ('root_quaternion_wxyz', (n, 4))]:
        if raw[key].shape != shape or not np.isfinite(raw[key]).all():
            raise ValueError('Invalid complete robot state array: ' + key)
    if not np.allclose(np.linalg.norm(raw['root_quaternion_wxyz'], axis=-1), 1., atol=1e-6, rtol=0):
        raise ValueError('Source root quaternion must already be normalized wxyz')
    target = load_target(row or source_robot(spec))
    model = target['model']
    import mujoco as mj
    if model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0 or any(t != mj.mjtJoint.mjJNT_HINGE for t in model.jnt_type[1:]):
        raise ValueError('Robot joint-state adapter requires one floating root and named hinge joints')
    joints = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
    expected = {model.joint(j).name for j in joints}
    if set(names.tolist()) != expected:
        raise ValueError('Source joint names must cover the local robot exactly; missing=' + str(sorted(expected-set(names))) + ', extra=' + str(sorted(set(names)-expected)))
    qpos = np.broadcast_to(model.qpos0, (len(take), model.nq)).copy()
    qpos[:, :3] = raw['root_position'][take]
    qpos[:, 3:7] = raw['root_quaternion_wxyz'][take]
    for index, name in enumerate(names):
        qpos[:, model.jnt_qposadr[model.joint(str(name)).id]] = raw['joint_positions'][take, index]
    return target, qpos, times, indices


def validate_attachment(package, manifest):
    from .dataset_io.storage import within
    from .motion_sources import sha256
    spec = dict(manifest['motion_source'])
    if any(key in spec for key in ('row','xml','model','model_path','model_url','decoder','command')):
        raise ValueError('Source robot models are local registry assets, not upload-selected paths')
    path = within(package, spec.get('path', ''))
    if not path.is_file() or path.suffix != '.npz':
        raise ValueError('Robot motion must be an NPZ inside the recording package')
    spec.update(path=str(path), family='robot', floor_z_m=spec.get('floor_z_m',manifest.get('floor_z_m')),
                evidence=spec.get('evidence','recorded'))
    _, _, times, indices = read_state(spec)
    if len(times) != manifest.get('frames', len(times)):
        raise ValueError('Robot motion and recording frame counts differ')
    # Uploads are already synchronized; no hidden slicing of one component.
    from .motion_sources import _npz
    if len(indices) != len(_npz(path)['timestamps_s']):
        raise ValueError('Upload robot motion must already match the recording window')
    if spec.get('objects_path'):
        objects = within(package, spec['objects_path'])
        if objects.suffix != '.npz' or not objects.is_file():
            raise ValueError('Robot objects_path must be a local package NPZ')
        spec['objects_path'] = str(objects)
        from .source_objects import load_source_objects
        load_source_objects(objects, times)
    spec['sha256'] = sha256(path)
    return spec


def prepare_robot_source(spec, output_dir, *, row=None, observed_work=None, source_manifest=None):
    import mujoco as mj
    from arenalab_motion.geometry import UniversalOracle, TASKS, BONE_ROLES
    from arenalab_motion.prior import palm_frame
    from greenwich_pipelines.viewer.export import visible_geoms, mesh_scene
    from greenwich_motion_sdk.motion import MotionClip
    from .motion_sources import sha256, _npz
    from .motion_backend import robot_asset_identity

    row = row or source_robot(spec)
    source_digest = sha256(spec['path'])
    target, qpos, times, indices = read_state(spec, row)
    model = target['model']; data = mj.MjData(model)
    semantics = row['semantic_metadata']['semantics']
    role_names = list(dict.fromkeys((*TASKS, *(r for pair in BONE_ROLES for r in pair))))
    semantic_ids = [model.body(semantics[role]).id for role in role_names]
    role_ids = [role_names.index(role) for role in TASKS]
    encoder_roles = []
    for role in TASKS:
        bid = model.body(semantics[role]).id
        while bid not in target['keep'] and bid > 0:
            bid = int(model.body_parentid[bid])
        if bid not in target['keep']:raise ValueError('Source semantic frame has no encoder ancestor')
        encoder_roles.append(int(np.flatnonzero(target['keep']==bid)[0]))
    # Do not rename robot joints to human names: the encoder keeps every node,
    # its original semantic name, physical rest geometry and native DoF features.
    names = list(target['spec'].joint_names)
    positions = np.empty((len(times), len(names), 3))
    rotations = np.empty((len(times), len(names), 3, 3))
    semantic_positions = np.empty((len(times), len(role_names), 3))
    semantic_rotations = np.empty((len(times), len(role_names), 3, 3))
    geoms = visible_geoms(model); geometry = mesh_scene(model, geoms)
    parts = [(g, geometry.geometry['geom_'+str(g)]) for g in geoms if 'geom_'+str(g) in geometry.geometry]
    if not parts:
        raise ValueError('Source robot requires visible geometry for its source preview')
    # Rigid robot links need one local mesh and per-frame poses, not a copy of
    # every vertex per frame. This display data is independent of the encoder.
    import trimesh
    visual = trimesh.Scene()
    for i, (_, mesh) in enumerate(parts):
        visual.add_geometry(mesh, node_name='geom_'+str(i), geom_name='geom_'+str(i))
    geom_positions = np.empty((len(times), len(parts), 3))
    geom_rotations = np.empty((len(times), len(parts), 3, 3))
    neutral_vertices = []
    mj.mj_forward(model, data)
    for g, mesh in parts:
        neutral_vertices.append(np.asarray(mesh.vertices) @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g])
    height = float(np.ptp(np.concatenate(neutral_vertices)[:,2]))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose; mj.mj_forward(model, data)
        positions[frame] = data.xpos[target['keep']]
        rotations[frame] = data.xmat[target['prior_body_ids']].reshape(-1,3,3)
        semantic_positions[frame] = data.xpos[semantic_ids]
        semantic_rotations[frame] = data.xmat[semantic_ids].reshape(-1,3,3)
        for i,(g,mesh) in enumerate(parts):
            geom_positions[frame,i] = data.geom_xpos[g]
            geom_rotations[frame,i] = data.geom_xmat[g].reshape(3,3)
    bones = np.stack([semantic_positions[:,role_names.index(b)]-semantic_positions[:,role_names.index(a)] for a,b in BONE_ROLES],1)
    lengths = np.linalg.norm(bones,axis=-1,keepdims=True)
    if np.any(lengths < 1e-8):
        raise ValueError('Degenerate source robot semantic limb')
    bones /= lengths
    six = semantic_positions[:,role_ids].copy(); reference_R = semantic_rotations[:,role_ids].copy()
    local = np.stack([palm_frame(row['palm_frames'][side+'_wrist']['distal'],row['palm_frames'][side+'_wrist']['normal']) for side in ('left','right')])
    palm = reference_R[:,2:4] @ local
    wrists = six[:,2:4].copy(); forward = palm[...,0]; normal = palm[...,2]
    oracle = UniversalOracle(row)
    _, soles = oracle.surface.exact_heights(qpos)
    floor = float(spec['floor_z_m'])
    common = dict(timestamps_s=times, source_frame_indices=indices, exact_fps=np.array(30.), nominal_fps=np.array(30.))
    observation_hashes = {}
    observed = Path(observed_work) if observed_work is not None else None
    if observed is not None and (observed/'observations.npz').is_file():
        obs = _npz(observed/'observations.npz')
        if obs['timestamps_s'].shape != times.shape or not np.allclose(obs['timestamps_s'],times,atol=1e-6,rtol=0):
            raise ValueError('Recorded operation and full robot motion clocks disagree')
        wrists,forward,normal = [np.asarray(obs[k],float) for k in ('wrist_position','palm_forward','palm_normal')]
        if any(v.shape!=(len(times),2,3) or not np.isfinite(v).all() for v in (wrists,forward,normal)):
            raise ValueError('Invalid independent robot operation observations')
        palm_frame(forward,normal)
        for name in ('observations.npz','operation-accuracy/source-targets.npz','source_hands.npz'):
            if (observed/name).is_file():observation_hashes[name]=sha256(observed/name)
    six[:,2:4] = wrists
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a fresh immutable robot source directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name+'.tmp-',dir=output.parent))
    try:
        p_y = positions@C.T; r_y = C@rotations@C.T
        np.savez_compressed(temp/'native_source.npz',**common,names=np.asarray(names),parents=target['spec'].parents,
            rest_offsets_cm=target['spec'].rest_offsets,dof=np.asarray(target['dof']),
            full_body_position_yup_m=p_y,full_body_rotation_yup=r_y,
            source_body_position_world_m=positions,source_body_rotation_world_transfer=rotations,
            semantic_names=np.asarray(role_names),semantic_position_world_m=semantic_positions,
            role_indices=np.asarray(encoder_roles), # Exact physical task views remain separate from merged encoder nodes.
            six_position_zup_m=six,reference_world_R_zup=reference_R,bone_directions_world=bones,
            wrist_position_world_m=wrists,palm_forward=forward,palm_normal=normal,
            sole_low_z_m=soles,floor_z_m=np.array(floor))
        (temp/'source_robot.glb').write_bytes(visual.export(file_type='glb'))
        np.savez_compressed(temp/'source_robot_playback.npz',**common,
            positions=geom_positions, rotations=geom_rotations)
        np.savez_compressed(temp/'observations.npz',**common,head_position=np.r_[six[:1,1],six[:,1]],head_rotation=np.concatenate([reference_R[:1,1],reference_R[:,1]]),
            wrist_position=wrists,palm_position=wrists,palm_forward=forward,palm_normal=normal,floor_z=np.array(floor))
        (temp/'operation-accuracy').mkdir()
        np.savez_compressed(temp/'operation-accuracy/source-targets.npz',points=wrists[:,:,None],timestamps_s=times)
        for name in observation_hashes:shutil.copyfile(observed/name,temp/name)
        if spec.get('objects_path'):
            from .source_objects import load_source_objects
            load_source_objects(spec['objects_path'], times)
            shutil.copyfile(spec['objects_path'],temp/'objects.npz')
        provenance=dict(source_manifest or {},schema='arenalab.motion-source.v1',policy=POLICY,sourcekind='supplied_robot',
            source_body='Supplied complete robot joint states; no human reconstruction',human_model=None,source_robot_id=spec['robot_id'],
            source_format=FORMAT,source_evidence=spec.get('evidence','recorded'),source_path=str(Path(spec['path']).resolve()),source_sha256=source_digest,
            source_model_sha256=target['model_sha256'],source_asset_sha256=robot_asset_identity(row['xml']),
            source_completion=False,neural_source_decode=False,generated_human_model=None,body_encoder_joints=len(names),
            frames=len(times),fps=30.,exact_fps=30.,source_frame_indices=indices.tolist(),neutral_mesh_height_m=height,
            source_pose_or_translation_modified=False,source_uniform_similarity_applied=False,source_world_up='z',source_units='m',
            floor_world_m=floor,floor_policy=spec.get('floor_policy','Declared source floor; no fitted replacement'),
            end_effector=spec.get('end_effector',(source_manifest or {}).get('end_effector','none')),
            source_support='Source local robot visual hull; exact mesh/primitive plane minima',
            task_reference_evidence='Independent recorded operation targets' if observation_hashes else 'Original robot wrist origins and declared palm axes',
            observed_source_sha256=observation_hashes,source_clock='Original selected 30 Hz samples; no interpolation',model_arrays_exported=False)
        if spec.get('video_role') is not None:provenance['source_video_role']=spec['video_role']
        MotionClip(names,target['spec'].parents,target['spec'].rest_offsets,r_y,p_y*100,times,provenance).save(temp/'canonical.npz')
        provenance.update(native_source_sha256=sha256(temp/'native_source.npz'),
            source_visual_format='rigid-link-poses-v1',source_model_glb_sha256=sha256(temp/'source_robot.glb'),
            source_visual_poses_sha256=sha256(temp/'source_robot_playback.npz'))
        provenance['output_sha256']={str(p.relative_to(temp)):sha256(p) for p in temp.rglob('*.npz')}
        for name in ('source_provenance.json','provenance.json','input_manifest.json'):
            (temp/name).write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
        if sha256(spec['path'])!=source_digest:raise RuntimeError('Original robot motion changed while adapting')
        if output.exists():output.rmdir()
        temp.rename(output)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    return output


def recorded_state_arrays(q, names, timestamps, row):
    """Map declared recording columns by name, including differently ordered hips.

    Only conventional `_joint` suffix equivalence is accepted; never infer an
    ordering from the number of columns or the robot/dataset/action label.
    """
    import mujoco as mj
    q=np.asarray(q,dtype=float); names=list(names); times=np.asarray(timestamps,dtype=float)
    if q.ndim!=2 or q.shape[1]!=len(names) or len(set(names))!=len(names) or not np.isfinite(q).all():
        raise ValueError('Recorded state needs finite values and unique declared column names')
    roots=['root_x','root_y','root_z','root_qw','root_qx','root_qy','root_qz']
    if not all(n in names for n in roots):raise ValueError('Recorded full-body root xyz/wxyz columns are required')
    model=mj.MjModel.from_xml_path(row['xml'])
    native=[model.joint(j).name for j in range(model.njnt) if model.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE]
    canonical=lambda name:name.removesuffix('_joint')
    columns=[n for n in names if n not in roots]
    if len(set(map(canonical,columns)))!=len(columns) or set(map(canonical,columns))!=set(map(canonical,native)):
        raise ValueError('Recorded joint column names do not uniquely match the local source robot')
    indices=[names.index(next(n for n in columns if canonical(n)==canonical(key))) for key in native]
    return dict(joint_names=np.asarray(native),joint_positions=q[:,indices],root_position=q[:,[names.index(n) for n in roots[:3]]],
                root_quaternion_wxyz=q[:,[names.index(n) for n in roots[3:]]],timestamps_s=times)


def write_recorded_source(q, names, timestamps, directory, robot_id, *, evidence='recorded', floor_z_m=0.):
    """Retain complete recorded states as a separate immutable source attachment."""
    spec=dict(format=FORMAT,family='robot',robot_id=robot_id,world_up='z',units='m',fps=30.,floor_z_m=float(floor_z_m),evidence=evidence)
    arrays=recorded_state_arrays(q,names,timestamps,source_robot(spec))
    digest=hashlib.sha256()
    for name,value in sorted(arrays.items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    output=Path(directory)/('robot-motion-'+digest.hexdigest()[:24]+'.npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    if not output.exists():
        with output.open('xb') as stream:np.savez_compressed(stream,**arrays)
    spec['path']=str(output.resolve())
    read_state(spec)
    from .motion_sources import sha256
    spec['sha256']=sha256(output)
    return spec
