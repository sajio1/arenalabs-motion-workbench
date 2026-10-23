"""LeRobot v3 tables/videos plus lossless ArenaLabs modalities and provenance.

No inference, dataset publication, or model-weight packaging happens here.
"""
from arenalab_workbench.settings import paths
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
import subprocess
import zipfile
import numpy as np
from .storage import STORE, within, read, write

VERSION = 'arenalabs-lerobot-v3.0-4-native-reference'


def asset_path(url):
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith('/assets/'):
        raise ValueError('Export accepts only local generated asset references')
    return within(STORE, parsed.path.removeprefix('/assets/'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def stats(value):
    value = np.asarray(value)
    return {k: v.tolist() for k, v in dict(min=value.min(0), max=value.max(0), mean=value.mean(0), std=value.std(0), count=np.array([len(value)])).items()}


def parquet(path, columns):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression='zstd')


def normalize_video(source, target, n, start, frame_times=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if frame_times is not None:
        import cv2
        import imageio.v2 as imageio
        from arenalab_workbench.video_io import read_at
        if len(frame_times)!=n:raise ValueError('Video clock does not cover the complete episode')
        cap=cv2.VideoCapture(str(source))
        try:
            with imageio.get_writer(target,fps=30,codec='libx264',quality=8,macro_block_size=1,ffmpeg_params=['-threads','2']) as writer:
                for seconds in frame_times:
                    ok,frame=read_at(cap,float(seconds))
                    if not ok:raise ValueError('Original video frame is unavailable at a source timestamp')
                    writer.append_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally:cap.release()
    else:
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', str(start), '-i', str(source), '-an', '-vf', 'fps=30', '-frames:v', str(n), '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', str(target)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(target)]))['streams'][0]
    if int(probe['nb_read_frames']) != n:
        raise ValueError('Video coverage differs from the robot episode; export was not finalized')
    return probe


def video_stats(path,n):
    import cv2
    cap=cv2.VideoCapture(str(path));selected=set(np.linspace(0,n-1,min(n,100)).astype(int).tolist())
    low=np.full(3,np.inf);high=np.full(3,-np.inf);total=np.zeros(3);square=np.zeros(3);pixels=0;samples=0
    try:
        for i in range(n):
            ok,image=cap.read()
            if not ok:raise ValueError('Video is truncated during statistics validation')
            if i not in selected:continue
            image=cv2.cvtColor(cv2.resize(image,(128,max(1,round(image.shape[0]*128/image.shape[1])))),cv2.COLOR_BGR2RGB).reshape(-1,3).astype(np.float64)/255
            low=np.minimum(low,image.min(0));high=np.maximum(high,image.max(0));total+=image.sum(0);square+=(image*image).sum(0);pixels+=len(image);samples+=1
    finally:cap.release()
    mean=total/pixels;std=np.sqrt(np.maximum(0,square/pixels-mean*mean))
    return {**{k:v.reshape(3,1,1).tolist() for k,v in dict(min=low,max=high,mean=mean,std=std).items()},'count':[samples]}


def native_reference(robot, motion, output, *, scene=None):
    """Bind native state to the exact producing model, never a current registry guess.

    The customer's model remains local. Only names, reference states and FK
    validation metadata are exported; no URDF, meshes or physical parameters.
    Missing legacy provenance returns an explicit unavailable status. The
    dataset readiness gate refuses to publish that as a complete new export.
    """
    validation = read(asset_path(robot['playback_url']).with_name('validation.json'), {})
    native = validation.get('native_output_dir')
    if not native:
        return dict(available=False, reason='No authoritative native output recorded; visual replay remains available')
    source = Path(native) / 'motion.npz'
    if not source.is_file():
        raise ValueError('Recorded native motion is missing; cannot certify this export')
    candidates = [source.resolve().parent / 'request.json']
    if validation.get('native_work_dir'):
        candidates.insert(0, Path(validation['native_work_dir']) / 'request.json')
    request_path = next((path for path in candidates if path.is_file()), None)
    if request_path is None:
        return dict(available=False, reason='Native state retained but producing model contract is unavailable')
    request = read(request_path)
    if request.get('robot_id') != robot['id']:
        raise ValueError('Native motion robot identity differs from selected export')
    if not request.get('output_dir') or Path(request['output_dir']).resolve() != source.resolve().parent:
        raise ValueError('Native producing request does not belong to this motion output')
    row = request.get('row', request.get('robot', {}))
    row = row.get('pipeline_row', row)
    model = row.get('xml') or row.get('model')
    if not model or not Path(model).is_file():
        raise ValueError('Producing robot model is unavailable; cannot certify native export')
    from .native_reference import build_native_reference
    result = build_native_reference(source, model, output, playback=motion, robot_id=robot['id'])
    summary = dict(available=True, contract='extensions/arenalabs/native_reference/manifest.json',
                mode='articulated_kinematic_reference', model_included=False,
                physics_control_validated=False, frames=result['frames'])
    if scene is not None:
        from .native_visual import validate_native_visual
        validate_native_visual(source, model, asset_path(robot['playback_url']), scene, robot, output)
        summary['visual_validation'] = 'extensions/arenalabs/native_reference/visual-validation.json'
    return summary


def export_dataset(scene_path, robot_id, output, raw_sources=(), video_sources=None):
    scene_path, output = Path(scene_path), Path(output)
    scene = read(scene_path)
    selected=scene.get('output_modalities')
    from arenalab_workbench.job_outputs import filtered_scene, STAGES
    if (not isinstance(selected,list) or not selected or
            any(not isinstance(name,str) or name not in STAGES for name in selected) or
            len(set(selected))!=len(selected)):
        raise ValueError('EXPORT_CURRENT_FORMAT_REQUIRED: new exports require an explicit valid output selection; regenerate legacy results')
    if 'trajectory' not in selected:
        raise ValueError('OUTPUT_TRAJECTORY_NOT_SELECTED: robot dataset export requires selected trajectory')
    completed=scene.get('completed_outputs')
    status=scene.get('output_status')
    generation=scene.get('generation_status')
    if (not isinstance(completed,list) or any(not isinstance(name,str) for name in completed) or
            len(set(completed))!=len(completed) or set(completed)!=set(selected) or
            not isinstance(status,dict) or any(status.get(name)!='completed' for name in selected) or
            not isinstance(generation,dict) or generation.get('status')!='completed'):
        raise ValueError('OUTPUTS_INCOMPLETE: every selected output must be recorded complete before export')
    # Validate the original completion evidence before filtering dependencies.
    # filtered_scene updates statuses; calling it first would relabel a failed
    # or pending selected output as complete. Existing ZIPs remain immutable.
    scene=filtered_scene(scene,{'effective_outputs':selected},
                         {STAGES[name]:status[name]=='completed' for name in selected})
    robot = next((r for r in scene['robots'] if r['id'] == robot_id), None)
    if robot is None:
        raise ValueError('Generate the selected robot before exporting')
    archive = output.with_suffix('.zip')
    pending = archive.with_suffix('.zip.partial')
    if any(path.exists() or path.is_symlink() for path in (output, archive, pending)):
        raise FileExistsError('Exports are immutable; use a fresh build directory')
    output.mkdir(parents=True)
    motion = read(asset_path(robot['playback_url']))
    n, fps = int(motion['frames']), float(motion['fps'])
    if fps != 30 or n != int(scene['frames']):
        raise ValueError('Scene and robot clocks differ')
    state = np.asarray(motion['joints']['values'], dtype='f4')
    if state.shape != (n, len(motion['joints']['names'])) or not np.isfinite(state).all():
        raise ValueError('Invalid robot joint states')
    action = np.concatenate([state[1:], state[-1:]])
    arrays = {'observation.state': state, 'action': action, 'timestamp': np.arange(n, dtype='f4') / 30, 'frame_index': np.arange(n, dtype='int64'), 'episode_index': np.zeros(n, dtype='int64'), 'index': np.arange(n, dtype='int64'), 'task_index': np.zeros(n, dtype='int64')}
    features = {k: dict(dtype=str(v.dtype), shape=[v.shape[1]] if v.ndim == 2 else [1], names=motion['joints']['names'] if v.ndim == 2 else None) for k, v in arrays.items()}
    import pyarrow as pa
    columns = {k:pa.array(v.tolist(),type=pa.list_(pa.from_numpy_dtype(v.dtype),v.shape[1])) if v.ndim==2 else pa.array(v) for k,v in arrays.items()}
    parquet(output / 'data/chunk-000/file-000.parquet', columns)
    task = scene.get('semantic', {}).get('task') or scene['title']
    import pandas as pd
    (output / 'meta').mkdir()
    pd.DataFrame({'task_index': [0]}, index=[task]).to_parquet(output / 'meta/tasks.parquet')
    statistics={k:stats(v) for k,v in arrays.items()}
    episode = dict(episode_index=[0], tasks=[[task]], length=[n], dataset_from_index=[0], dataset_to_index=[n], **{'data/chunk_index': [0], 'data/file_index': [0], 'meta/episodes/chunk_index': [0], 'meta/episodes/file_index': [0]})
    video_sources=video_sources or {}
    original_video=video_sources.get('original',scene.get('video'))
    videos = [('original', original_video)] if original_video else []
    videos += [(f'wrist_{v["side"]}', v) for v in scene.get('wrist_videos', []) if v]
    if scene.get('patch_videos', {}).get(robot_id):
        videos.append(('inpaint', dict(url=scene['patch_videos'][robot_id], start_seconds=0)))

    def video_job(item):
        name, video = item
        key = 'observation.images.' + name
        path = output / f'videos/{key}/chunk-000/file-000.mp4'
        source=Path(video['path']) if video.get('path') else asset_path(video['url'])
        probe = normalize_video(source, path, n, float(video.get('start_seconds', 0)),video.get('frame_times'))
        return key, probe, video_stats(path,n)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for key, probe, image_stats in pool.map(video_job, videos):
            statistics[key]=image_stats
            features[key] = dict(dtype='video', shape=[probe['height'], probe['width'], 3], names=['height', 'width', 'channels'], info={'video.fps': 30, 'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False})
            for suffix, value in [('chunk_index', 0), ('file_index', 0), ('from_timestamp', 0.), ('to_timestamp', n / 30)]:
                episode[f'videos/{key}/{suffix}'] = [value]
    for key,values in statistics.items():
        for name,value in values.items():episode[f'stats/{key}/{name}']=[value]
    write(output/'meta/stats.json',statistics)
    parquet(output / 'meta/episodes/chunk-000/file-000.parquet', episode)
    write(output / 'meta/info.json', dict(codebase_version='v3.0', robot_type=robot_id, total_episodes=1, total_frames=n, total_tasks=1, total_videos=len(videos), total_chunks=1, chunks_size=1000, data_files_size_in_mb=100, video_files_size_in_mb=200, fps=30, splits={'train': '0:1'}, data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet', video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4', features=features))
    extension = output / 'extensions/arenalabs'
    extension.mkdir(parents=True)
    copied = {}

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items() if k not in {'owner'}}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and value.startswith('/assets/'):
            original = asset_path(value)
            if not original.exists():
                raise ValueError('Referenced modality is missing: ' + original.name)
            if original not in copied:
                target = extension / 'assets' / (hashlib.sha256(str(original).encode()).hexdigest()[:12] + '-' + original.name)
                target.parent.mkdir(exist_ok=True)
                if original.is_dir():shutil.copytree(original,target,ignore=shutil.ignore_patterns('.artifact-revisions'))
                else:shutil.copyfile(original, target)
                copied[original] = target.relative_to(output).as_posix()
            return copied[original]
        if isinstance(value, str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            return Path(value.replace('\\', '/')).name
        return value

    snapshot = {**scene, 'robots': [robot], 'patch_videos': {robot_id: scene.get('patch_videos', {}).get(robot_id)} if scene.get('patch_videos', {}).get(robot_id) else {}}
    if original_video:
        native_video=Path(original_video['path']) if original_video.get('path') else asset_path(original_video['url'])
        native_target=extension/'assets/original-native.mp4';native_target.parent.mkdir(exist_ok=True);shutil.copyfile(native_video,native_target)
        snapshot['video']=dict(url='videos/observation.images.original/chunk-000/file-000.mp4',start_seconds=0,native_original=native_target.relative_to(output).as_posix(),source_clock=original_video.get('frame_times'),source_start_seconds=original_video.get('start_seconds',0))
    else:snapshot['video']=None
    for field in ['patch_kinds','patch_depth_policy','patch_unavailable','hand_motion_audit_urls']:
        if field in snapshot:snapshot[field]={robot_id:snapshot[field][robot_id]} if robot_id in snapshot[field] else {}
    write(extension / 'scene.json', portable(snapshot))
    from .replay import build_replay
    replay = build_replay(asset_path(robot['model_url']), motion, extension/'replay')
    native = native_reference(robot, motion, extension/'native_reference', scene=scene)
    if not native['available']:
        raise ValueError('NATIVE_REFERENCE_UNAVAILABLE: ' + native['reason'])
    # Full raw masks, native depth/camera predictions and human motion stay lossless.
    raw_index = []
    for category, folder in raw_sources:
        if selected is not None and 'point_cloud' not in selected and category in {'perception','calibration'}:continue
        folder = Path(folder)
        if folder.is_file():
            target=extension/'raw'/category/'lens-rectification.npz';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder,target);raw_index.append(target.relative_to(output).as_posix());continue
        allowed = ['depth.npz', 'depth.json', 'frames.npz', 'cameras.npz', 'camera-intrinsics.npz', 'preparation.json', 'prompts.json', 'semantic.json', 'observations.npz', 'source_hands.npz', 'canonical.npz', 'egoallo_smplh.npz', 'smplh_motion.npz', 'human_reference.json', 'human_validation.json', 'prior-ready.json', 'source.npz', 'source.json', 'input_manifest.json', 'inference_identity.json', 'camera-calibration.json', 'lens-calibration.json', 'lens-rectification.npz', 'motion.npz', 'prime_original.npz', 'tool.npz', 'wuji_angles.npz', 'provenance.json', 'attachment.json', 'hand-mounts.json', 'hand-alignment.json', 'hand-alignment-correspondences.npz', 'wheeled-motion.npz', 'unrefined-motion.npz', 'wheeled-validation.json', 'encoder-input.json', 'report.json', 'validation.json']
        paths = [folder / name for name in allowed if (folder / name).is_file()]
        if (folder/'depth-store.json').is_file():
            paths.append(folder/'depth-store.json')
            relative=Path(read(folder/'depth-store.json')['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid depth store path')
            runs={(folder/relative).parent}
            calibration=read(folder/'camera-calibration.json') if (folder/'camera-calibration.json').exists() else {}
            bootstrap=calibration.get('prior_depth_receipt',{}).get('path')
            if bootstrap:
                prior=Path(bootstrap)
                if prior.is_absolute() or '..' in prior.parts:raise ValueError('Invalid calibration prior path')
                runs.add((folder/prior).parent)
            for run in sorted(runs):paths += sorted(p for p in run.rglob('*') if p.is_file())
        paths += sorted((folder / 'masks').glob('*-masks.npz'))
        if (folder/'masks/segmentation.json').is_file():paths.append(folder/'masks/segmentation.json')
        for path in paths:
            target = extension / 'raw' / category / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix=='.json':write(target,portable(read(path)))
            else:shutil.copyfile(path, target)
            raw_index.append(target.relative_to(output).as_posix())
    # Preserve the exact producing native motion even for direct callers that
    # did not collect optional raw-source folders. Never mask a misrouted raw
    # category by overwriting a different motion with the correct one.
    native_manifest=read(extension/'native_reference/manifest.json')
    raw_motion=extension/'raw/robot/refined/motion.npz'
    expected_native_sha=native_manifest['source']['sha256']
    if raw_motion.exists():
        if digest(raw_motion)!=expected_native_sha:
            raise ValueError('RAW_NATIVE_REFERENCE_MISMATCH: collected raw motion belongs to a different native result')
    else:
        validation=read(asset_path(robot['playback_url']).with_name('validation.json'),{})
        authoritative=Path(validation['native_output_dir'])/'motion.npz'
        if not authoritative.is_file() or digest(authoritative)!=expected_native_sha:
            raise ValueError('RAW_NATIVE_REFERENCE_MISMATCH: producing motion changed during packaging')
        raw_motion.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(authoritative,raw_motion)
        if digest(raw_motion)!=expected_native_sha:
            raise ValueError('RAW_NATIVE_REFERENCE_MISMATCH: copied native motion changed during packaging')
    raw_index=sorted(set(raw_index+[raw_motion.relative_to(output).as_posix()]))
    cloud = scene.get('cloud', {}).get('frames', [])
    clouds = {int(f['source_frame']): i for i, f in enumerate(cloud)}
    keys = sorted(clouds)
    frame_rows = []
    nearest=np.searchsorted(keys,np.arange(n),side='right')-1
    for frame in range(n):
        selected = keys[nearest[frame]] if nearest[frame]>=0 else None
        frame_rows.append(dict(frame_index=frame, timestamp=frame / 30, cloud_index=clouds[selected] if selected is not None else None, cloud_source_frame=selected, cloud_observed_this_frame=selected == frame))
    parquet(extension / 'frame_map.parquet', {k: [r[k] for r in frame_rows] for k in frame_rows[0]})
    shutil.copyfile(Path(__file__).with_name('portable_reader.py'), extension/'read_extensions.py')
    write(extension / 'manifest.json', dict(schema=VERSION, scene_id=scene['id'], robot_id=robot_id, fps=30, frames=n, state_units=motion['joints'].get('units', motion['joints'].get('unit')), action_semantics='Next-frame retargeted kinematic joint-position reference; final frame repeats. Not measured actuation or executed success.', state_semantics='Generated kinematic playback, not measured robot telemetry', contact_semantics='Retain original status, fps, evidence and nullable per-digit estimates; not measured pressure unless explicitly declared', frame_map='extensions/arenalabs/frame_map.parquet', raw_files=raw_index, missing={k: reason for k, reason in {'point_cloud': None if scene.get('cloud') else 'Not generated', 'inpaint_fov': scene.get('patch_unavailable', {}).get(robot_id,'Not generated') if not scene.get('patch_videos', {}).get(robot_id) else None, 'contact': scene.get('tactile', {}).get('reason') if not scene.get('tactile', {}).get('url') else None}.items() if reason}, source_scene_sha256=digest(scene_path), models_included=False, robot_visual_asset_included=True, model_weights_included=False, replay=dict(stage='extensions/arenalabs/replay/robot.usda', contract='extensions/arenalabs/replay/replay.json', mode=replay['mode'], physics_articulation=False)))
    manifest = read(extension / 'manifest.json')
    manifest['native_reference'] = native
    from .readiness import _modality_contract, validate_dataset
    manifest['modalities'] = _modality_contract(output.resolve(), read(extension / 'scene.json'), manifest, robot_id)
    manifest['readiness'] = dict(status='ready', scope='complete_kinematic_reference_data',
                                validator='validate_dataset.py', dynamics_validated=False)
    write(extension / 'manifest.json', manifest)
    shutil.copyfile(Path(__file__).with_name('readiness.py'), output / 'validate_dataset.py')
    hashes = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob('*') if p.is_file()}
    write(output / 'checksums.json', hashes)
    readiness = validate_dataset(output)
    with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                zipped.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(pending) as zipped:
        if zipped.testzip() is not None or set(zipped.namelist()) != set(hashes) | {'checksums.json'}:
            raise ValueError('Final ZIP integrity check failed; export is not ready')
    pending.rename(archive)
    return dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest(archive), frames=n, robot_id=robot_id, format=VERSION, ready=True, readiness=readiness)


def collect_raw_sources(scene_id, scene, robot_id=None):
    from .storage import ROOT
    from arenalab_workbench.gripper_registry import sources
    result=[]
    if scene_id in sources():
        from arenalab_workbench.gripper_body import body_path
        work=Path(sources()[scene_id]['work']);body=body_path(work);result += [('source',str(work)),('human',str(body))]
        gripper_work=work
    else:
        from arenalab_workbench.body_prior import folder
        body=folder(scene_id);result.append(('human',str(body)));gripper_work=None
    if robot_id is None and len(scene.get('robots',[]))==1:robot_id=scene['robots'][0]['id']
    if robot_id:
        selected=next((r for r in scene.get('robots',[]) if r['id']==robot_id),{})
        validation=read(asset_path(selected['playback_url']).with_name('validation.json'),{}) if selected.get('playback_url') else {}
        if gripper_work:
            refined=Path(validation['native_output_dir']) if validation.get('native_output_dir') else gripper_work/robot_id
            result.append(('robot/refined',str(refined)));native=body
        else:
            native=body/'retarget' if (body/'retarget'/(robot_id+'-result')).exists() else body/'retarget'/robot_id
            if validation.get('native_work_dir'):native=Path(validation['native_work_dir'])
            refined=Path(validation['native_output_dir']) if validation.get('native_output_dir') else native/'operation-accuracy/palm'/robot_id
            result.append(('robot/refined',str(refined)))
        for path in (native/(robot_id+'-result')).rglob('prime_original.npz'):
            result.append(('robot/original',str(path.parent)))
        from arenalab_workbench.wheeled_body import POLICY as MOBILE_POLICY
        mobile=body/'mobile'/MOBILE_POLICY/robot_id
        if mobile.exists():result.append(('robot/mobile',str(mobile)))
    scene_work=scene.get('geometry',{}).get('scene_work')
    if scene_work:
        path=Path(scene_work);result.append(('perception',str(path if path.is_absolute() else ROOT/'scene-jobs'/path)))
    if scene.get('dataset')=='Nymeria':
        from arenalab_workbench.scene_job import resolve_clip
        _,row=resolve_clip(scene_id);sequence=row.get('metadata',{}).get('Sequence',Path(row['source']).parent.name)
        profile=ROOT/'nymeria-calibration'/(sequence+'.npz')
        if profile.is_file():result.append(('calibration',str(profile)))
    return result


def collect_video_sources(scene_id,scene):
    from arenalab_workbench.gripper_registry import sources
    if scene_id in sources():
        meta=read(Path(sources()[scene_id]['work'])/'source.json')
        return {'original':dict(path=meta['head_video'],start_seconds=meta.get('start_seconds',0))}
    from arenalab_workbench.scene_job import resolve_clip
    _,row=resolve_clip(scene_id)
    return {'original':dict(row['video'])} if row.get('video') else {}
