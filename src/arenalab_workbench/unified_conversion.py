"""Motion Lab worker: provided SMPL or three-point H -> one resident GPU refiner.

Runs under the shared GPU lease. This entry has no legacy 100+100 refinement,
model download, body-family fallback, or tolerance experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
from arenalab_workbench.settings import paths
from arenalab_workbench.motion_backend import normalized_source,prepare_gpu_version,run_gpu_motion
from greenwich_pipelines.viewer.export import array_json,motion_payload,join_playback_segments

C=np.array([[0.,1,0],[0,0,1],[1,0,0]])


def three_point_work(clip, role_indices, output):
    """Keep actual source head/wrist poses; omit hidden full-body trajectories."""
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    roles=list(role_indices); p=np.asarray(clip.world_position_cm)/100 @ C
    r=C.T @ np.asarray(clip.global_rotation) @ C
    times=np.asarray(clip.timestamps_s)
    head=roles[1];wrists=roles[2:4]
    wp=p[:,wrists];wr=r[:,wrists]
    floor=float(clip.metadata.get('ground_y_cm',0.))/100
    np.savez_compressed(output/'observations.npz',head_position=np.r_[p[:1,head],p[:,head]],
        head_rotation=np.concatenate([r[:1,head],r[:,head]]),wrist_position=wp,palm_position=wp,
        palm_forward=wr[...,2],palm_normal=wr[...,0],timestamps_s=times,floor_z=np.array(floor))
    (output/'operation-accuracy').mkdir(exist_ok=True)
    np.savez_compressed(output/'operation-accuracy/source-targets.npz',points=wp[:,:,None])
    array_json(output/'input_manifest.json',dict(clip=output.name,dataset='Motion source',end_effector='none',
        frames=len(times),fps=30,human_model='smplh',source_policy='three-point-egoallo',source_evidence=clip.metadata,
        source_sha256=hashlib.sha256(p.tobytes()+r.tobytes()+times.tobytes()).hexdigest(),source_trajectory_edited=False))
    for name in ('model.npz','egoallo_checkpoint_april13'):
        asset=paths.egoallo/name
        if not asset.exists():raise FileNotFoundError(asset)
        if not (output/name).exists():(output/name).symlink_to(asset,target_is_directory=asset.is_dir())
    from arenalab_workbench.motion_sources import validate_three_point_observations
    validate_three_point_observations(output)
    from arenalab_workbench.body_inference import run
    run(output)
    return normalized_source(output)


def _source_options(manifest_path, raw):
    """Use the same explicit frame/profile options as the source SDK."""
    from greenwich_motion_sdk.source_batch import resolve
    options = {}
    profile_base = manifest_path.parent
    if raw.get('options_file'):
        profile = resolve(manifest_path.parent, raw['options_file'])
        options = json.loads(profile.read_text())
        profile_base = profile.parent
    options.update(raw.get('options', {}))
    if 'bind_path' in options:
        options['bind_path'] = resolve(profile_base, options['bind_path'])
    return options


def _provided(spec, directory, base=None):
    from arenalab_workbench.motion_sources import prepare_supplied_source
    spec = dict(spec)
    for key in ('path', 'objects_path'):
        if key in spec and base is not None and not Path(spec[key]).is_absolute():
            spec[key] = str((base / spec[key]).resolve())
    model = None if spec.get('format')=='robot_joint_state' else os.environ.get('ARENA_' + spec['family'].upper() + '_MODEL')
    return prepare_supplied_source(spec, directory, model_path=model)


def _smpl_manifest_spec(path, options, clip):
    """Read parameters/clock directly; do not run the legacy skeleton adapter."""
    from types import SimpleNamespace
    from greenwich_motion_sdk.source_batch import sample_clip
    with np.load(path, allow_pickle=False) as d:
        n = len(d['poses'])
        width = np.asarray(d['poses']).reshape(n, -1).shape[1]
        family = {72: 'smpl', 156: 'smplh', 165: 'smplx'}.get(width)
        if family is None:
            raise ValueError('Declare an explicit SMPL-family motion_source for this pose layout')
        fps = float(d.get('mocap_framerate', d.get('mocap_frame_rate', d.get('fps', 0))))
        times = np.asarray(d['timestamps_s'], float) if 'timestamps_s' in d else np.arange(n) / fps if fps > 0 else None
    if times is None or times.shape != (n,) or not np.isfinite(times).all():
        raise ValueError('Supplied SMPL requires an explicit finite source clock')
    if fps <= 0 and n > 1:
        fps = 1. / float(np.median(np.diff(times)))
    take, _ = sample_clip(SimpleNamespace(timestamps_s=times), 30,
                          clip.get('max_seconds') or (times[-1] - times[0] + 1 / fps))
    frame = options.get('frame', {})
    basis = np.asarray(frame.get('basis_to_yup', np.eye(3)))
    if not np.allclose(basis, np.eye(3)) and not np.allclose(basis, C):
        raise ValueError('SMPL source needs an explicit y/z world basis in motion_source')
    cm_per_unit = float(frame.get('cm_per_unit', 100))
    if cm_per_unit not in (1., 100.):
        raise ValueError('Declare metres or centimetres for a supplied SMPL source')
    return dict(format='smpl_parameters', family=family, path=str(path),
                world_up='y' if np.allclose(basis, np.eye(3)) else 'z',
                units='m' if cm_per_unit == 100 else 'cm', evidence='fitted',
                floor_z_m=float(frame.get('ground_y_cm', 0)) / 100, fps=fps,
                start_frame=int(take[0]), stride=int(take[1]-take[0]), frames=len(take), end_effector='none')


def prepare_sources(clip, output):
    """Supplied SMPL-family stays supplied; absent SMPL body uses three points."""
    output = Path(output)
    supplied = clip.get('motion_source') or (clip.get('manifest') or {}).get('motion_source')
    if supplied:
        return [_provided(supplied, output / 'source')]
    source = Path(clip['source']).resolve()
    if source.suffix.lower() != '.json':
        # Registered ego/real-robot sources have an explicit three-point adapter.
        from arenalab_workbench.body_prior import run
        return [normalized_source(run(clip['id']))]
    from greenwich_motion_sdk.source_batch import resolve, sample_clip
    from greenwich_motion_sdk.adapters import load_motion
    from greenwich_motion_sdk.motion import CoordinateFrame, MotionClip
    manifest = json.loads(source.read_text())
    if manifest.get('schema') != 'alphamotion.sources.v1' or not manifest.get('sources'):
        raise ValueError('Expected nonempty alphamotion.sources.v1 manifest')
    result = []; seen = set()
    for index, raw in enumerate(manifest['sources']):
        identifier = raw['id']
        if not identifier or identifier in seen or '/' in identifier or '\\' in identifier or identifier in ('.', '..'):
            raise ValueError('Source IDs must be unique safe basenames')
        seen.add(identifier)
        directory = output / f'source-{index:03d}'
        path = resolve(source.parent, raw['path'])
        options = _source_options(source, raw)
        if raw.get('motion_source'):
            result.append(_provided(raw['motion_source'], directory, source.parent)); continue
        if raw['format'] == 'smpl':
            result.append(_provided(_smpl_manifest_spec(path, options, clip), directory)); continue
        if 'frame' in options:
            options['frame'] = CoordinateFrame(**options['frame'])
        data = load_motion(path, format=raw['format'], **options)
        # Only explicit, traceable parameter provenance can bypass reconstruction.
        if data.metadata.get('motion_source'):
            result.append(_provided(data.metadata['motion_source'], directory, path.parent)); continue
        take, _ = sample_clip(data, 30, clip.get('max_seconds') or (data.timestamps_s[-1]-data.timestamps_s[0]+1/30))
        roles = raw.get('roles')
        if raw['format'] == 'soma':
            roles = roles or ['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']
        if not roles or len(roles) != 6:
            raise ValueError('Declare root/head/L-wrist/R-wrist/L-ankle/R-ankle source roles')
        role_indices = [data.names.index(name) for name in roles]
        if role_indices[0] != 0:
            raise ValueError('The source root role must be the hierarchy root')
        part = MotionClip(data.names, data.parents, data.rest_offsets_cm,
                          data.global_rotation[take], data.world_position_cm[take], data.timestamps_s[take], data.metadata)
        result.append(three_point_work(part, role_indices, directory))
    return result


def run(request):
    output=Path(request['output']).resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ['ARENA_MOTION_RUNTIME']=str(Path(request['runtime']).resolve())
    os.environ['ARENA_TRAJECTORY_SCALE']=request.get('trajectory_scale','preserve')
    os.environ.pop('ARENA_STREAM_ROBOT_ID',None)
    robot=request['robot'];row=dict(robot.get('pipeline_row',robot))
    from arenalab_motion.robot_metadata import ensure_robot_metadata
    row=ensure_robot_metadata(row)
    from alphamotion.engine.descriptor import build_from_mjcf
    from greenwich_pipelines.robot_semantics import resolve_endpoints,ROLES
    spec,*_=build_from_mjcf(row['xml'],row['body']);ids=resolve_endpoints(spec,row.get('endpoints'))
    row['endpoints']={role:spec.joint_names[i] for role,i in zip(ROLES,ids)}
    segments=[];reports=[]
    for index,source in enumerate(prepare_sources(request['clip'],output)):
        with np.load(source/'native_source.npz',allow_pickle=False) as d:n=len(d['timestamps_s'])
        work,job,cached=prepare_gpu_version(output/f'motion-{index:03d}',robot.get('id',row['body']),row,source,'tcp',np.zeros((n,2,1,3)))
        if cached is None:report,_=run_gpu_motion(job,cwd=work)
        else:report,_=cached
        motion=Path(job['output_dir'])/'motion.npz'
        payload=motion_payload(row,motion,'fullbody')
        with np.load(motion,allow_pickle=False) as d, np.load(motion.with_name('source-targets.npz'),allow_pickle=False) as t:
            actual=d['wrist_body_position_native_m']@C.T;target=t['points'][:,:,0]@C.T
            # Display actual source targets, not an alias of the robot skeleton.
            payload['targets']=target.tolist()
            payload['tracking']=dict(schema='arenalab.tracking.v1',thresholds=dict(position_mm=10,orientation_deg=15),roles=['left_wrist','right_wrist'],reference='Declared source wrist vs final native FK',unit='m',axis='y-up',
                actual_positions=actual.tolist(),target_positions=target.tolist(),position_error_mm=(np.linalg.norm(actual-target,axis=-1)*1000).tolist(),orientation_error_deg=d['wrist_error_deg'].tolist())
        payload.update(trajectory_scale=job['scale_mode'],pipeline_version='resident-skeleton-surface-wrist-v3',solver=report.get('solver'),source_transform=report.get('scale_transform'))
        segments.append(payload);reports.append(report)
    metadata=[dict(frames=p['frames'],source_transform=p.get('source_transform'),solver=p.get('solver')) for p in segments]
    playback=join_playback_segments(segments)
    playback['segment_metadata']=metadata
    if any(m['source_transform']!=metadata[0]['source_transform'] for m in metadata[1:]):
        playback.pop('source_transform',None)
    array_json(output/'playback.json',playback)
    array_json(output/'report.json',dict(pipeline_version='resident-skeleton-surface-wrist-v3',trajectory_scale=request.get('trajectory_scale','preserve'),physics_status='not_run',source_policy=reports[0].get('source_policy') if len(reports)==1 else 'multiple-sources',source_policy_version='unified-provided-body-or-three-point-native-h-v1',solver_status=[r.get('solver') for r in reports],segments=reports))
    return playback


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.request.read_text()))
