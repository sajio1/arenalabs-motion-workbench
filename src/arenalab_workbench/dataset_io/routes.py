"""Thin HTTP layer: streamed imports and asynchronous, immutable dataset downloads."""
from arenalab_workbench.settings import paths
from pathlib import Path
import concurrent.futures
import json
import os
import secrets
import shutil
import subprocess
import uuid
import zipfile
import numpy as np
from fastapi import File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from . import storage as s
from .ingest import MAX_ARCHIVE, unpack, validate_normalized


class ExportRequest(BaseModel):
    scene_id: str
    robot_id: str
    trajectory_scale: Literal['preserve','match_robot']|None = None
    output_modalities: list[Literal['trajectory','point_cloud','contact','inpaint_fov']]|None = Field(default=None,min_length=1,max_length=4)

    @model_validator(mode='after')
    def unique_outputs(self):
        if self.output_modalities is not None:
            from arenalab_workbench.output_modalities import OUTPUTS
            if len(set(self.output_modalities))!=len(self.output_modalities):raise ValueError('Output modalities must be unique')
            self.output_modalities=[name for name in OUTPUTS if name in self.output_modalities]
        return self


def select_export_scene(body, scene_id, public):
    """Select the exact generated variant; explicit outputs never fall back."""
    from .export import asset_path
    from arenalab_workbench.unified_scene import scene_key
    from arenalab_workbench.output_modalities import resolve_outputs
    def rejected(code,message):
        raise HTTPException(409,dict(code=code,message=message))
    def modality_error():
        rejected('OUTPUT_TRAJECTORY_NOT_SELECTED','Robot dataset export requires trajectory among the selected outputs; point-cloud, contact, or FOV-only results are not robot datasets')
    if body.output_modalities is not None and 'trajectory' not in body.output_modalities:modality_error()
    public_path=asset_path(public['scene_url']) if public and public.get('scene_url') else None
    mode=body.trajectory_scale or 'preserve'
    def selected(scene):
        return scene.get('output_modalities',scene.get('generation_status',{}).get('output_modalities'))
    def matching_robot(scene):
        return any(row.get('id')==body.robot_id and row.get('trajectory_scale',scene.get('generation_status',{}).get('trajectory_scale','preserve'))==mode for row in scene.get('robots',[]))
    path=None;expected=None
    if body.trajectory_scale is None and body.output_modalities is None and public_path and public_path.is_file():
        path=public_path
    else:
        expected=body.output_modalities
        if expected is None:
            from arenalab_workbench.source_requirements import input_requirements
            try:expected=resolve_outputs(None,input_requirements(scene_id)['output_capabilities'])['effective_outputs']
            except (ValueError,OSError):expected=None
        if expected is not None:
            if 'trajectory' not in expected:modality_error()
            candidate=s.STORE/'manipulation4d/unified'/scene_key(scene_id,mode,body.robot_id,expected)/'scene.json'
            if candidate.is_file():path=candidate
        if path is None and body.output_modalities is None:
            # Compatibility applies only to actual pre-selector scenes, with an
            # exact robot/scale match. Never substitute another selected variant.
            legacy=s.STORE/'manipulation4d/unified'/scene_key(scene_id,mode,body.robot_id)/'scene.json'
            for candidate in (legacy,public_path):
                if candidate is None or not candidate.is_file():continue
                old=s.read(candidate)
                if old.get('id')==scene_id and selected(old) is None and matching_robot(old):path=candidate;expected=None;break
    if path is None:rejected('OUTPUT_VARIANT_UNAVAILABLE','Generate the requested output and trajectory-scale variant before downloading')
    scene=s.read(path)
    if scene.get('id')!=scene_id:rejected('OUTPUT_VARIANT_MISMATCH','Generated scene belongs to a different recording')
    modalities=selected(scene)
    if modalities is not None and 'trajectory' not in modalities:modality_error()
    if expected is not None and (modalities is None or set(modalities)!=set(expected)):
        rejected('OUTPUT_VARIANT_MISMATCH','Generated scene does not match the selected output variant')
    if modalities is not None and 'output_status' in scene and any(scene['output_status'].get(name)!='completed' for name in modalities):
        rejected('OUTPUTS_INCOMPLETE','Selected outputs are still being generated')
    if not any(row.get('id')==body.robot_id and (body.trajectory_scale is None or row.get('trajectory_scale',scene.get('generation_status',{}).get('trajectory_scale','preserve'))==body.trajectory_scale) for row in scene.get('robots',[])):
        rejected('ROBOT_RESULT_UNAVAILABLE','Generate the selected robot for this variant before downloading')
    return path,scene


def require_scene(scene_id, request):
    if not s.authorize_scene(scene_id, request.cookies.get(s.COOKIE)):
        raise HTTPException(404, 'Recording is unavailable in this workspace')


def import_recording(archive, ident, workspace):
    """Register a source only after all archive, schema and clock validation passes."""
    from arenalab_workbench.gpu_scheduler import resource_lock
    from .export import digest as file_digest
    work = s.ROOT / 'user-data' / ident
    package = work / 'package'
    unpack(archive, package)
    roots = list(package.rglob('recording.json'))
    if len(roots) != 1:
        raise ValueError('Include exactly one recording.json; download the input template')
    package = roots[0].parent
    manifest, arrays, video = validate_normalized(package)
    dest = s.STORE / 'manipulation4d/uploads' / ident
    s.mark_private(dest, workspace)
    public_video = None
    if video is not None:
        public_video = dest / 'source/original.mp4'
        public_video.parent.mkdir(exist_ok=True)
        stream=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt','-of','json',str(video)]))['streams'][0]
        if stream.get('codec_name')=='h264' and stream.get('pix_fmt') in {'yuv420p','yuvj420p'}:shutil.copyfile(video, public_video)
        else:
            # Ubuntu's installed FFmpeg can predate -fps_mode. -vsync 0 has the
            # same timestamp-preserving behavior across the supported binaries.
            encoded=subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-i',str(video),'-map','0:v:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-vsync','0','-movflags','+faststart',str(public_video)],capture_output=True)
            if encoded.returncode:
                raise ValueError('Video could not be encoded for browser playback; check the video codec and frames')
    mode = manifest['end_effector']
    n = len(arrays['timestamps_s'])
    np.savez_compressed(work / 'observations-normalized.npz', **arrays)
    video_url = '/assets/' + public_video.relative_to(s.STORE).as_posix() if public_video else None
    public = dict(id=ident, title=str(manifest.get('title', 'Uploaded recording'))[:160], dataset='Uploaded', end_effector=mode, family='My recordings', duration=(n-1)/30, owner=workspace, uploaded=True, video=dict(url=video_url, start_seconds=float(manifest.get('video_start_seconds', 0))))
    if video is None:
        public['video'] = None
    supplied = manifest.get('_resolved_motion_source')
    if supplied is not None:
        public.update(has_supplied_body=True, source_body_family=supplied['family'], source_evidence=supplied['evidence'], motion_source_policy='supplied-robot-fullbody-v1' if supplied['format']=='robot_joint_state' else 'supplied-body22-surface-v1')
    source_row = dict(id=ident, source=str(work / 'observations-normalized.npz'), frames=n, fps=30, video=dict(path=str(video), start_seconds=public['video']['start_seconds']) if video is not None else None, metadata=dict(upload_schema=manifest['schema']))
    record = dict(public=public, source_row=source_row, manifest=manifest, destination=str(dest), archive_sha256=file_digest(archive))
    if supplied is not None:
        record['motion_source'] = supplied
        record['body_adapter'] = 'supplied-robot' if supplied['format']=='robot_joint_state' else 'supplied-body'
        source_row['observations_source'] = source_row['source']
    if mode == 'robotiq-2f-140':
        source = dict(camera=arrays['camera'], intrinsic=arrays['intrinsic'], tcp_zup=arrays['wrist_position'], tool_R_zup=arrays['ee_rotation'], width_m=arrays['width_m'], floor_z=arrays['floor_z'], timestamps_s=arrays['timestamps_s'])
        np.savez_compressed(work / 'source.npz', **source)
        s.write(work / 'source.json', dict(dataset='Uploaded', frames=n, fps=30, task=public['title'], head_video=str(video), start_seconds=public['video']['start_seconds'], wrist_videos=[], target_reference=manifest.get('ee_reference', 'User-declared fixed gripper housing frame'), width_provenance='User-declared gripper width in metres', floor_provenance='User-declared floor_z_m', source_sha256=record['archive_sha256'], source_trajectory_edited=False, camera_image_size=manifest['image_size']))
        if supplied is not None:
            metadata=s.read(work/'source.json');metadata['motion_source']=supplied
            s.write(work/'source.json',metadata)
        record['gripper'] = dict(work=str(work), destination=str(dest))
    elif mode == 'wuji':
        # Convert declared source observations to the established EgoDex schema;
        # preserve every observed landmark, with only a documented basis change.
        import h5py
        c = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        hdf = work / 'egodex-observations.hdf5'
        with h5py.File(hdf, 'w') as f:
            poses = arrays['camera'].copy()
            poses[:, :3, :3] = c.T @ poses[:, :3, :3]
            poses[:, :3, 3] = poses[:, :3, 3] @ c
            f['transforms/camera'] = poses
            f['camera/intrinsic'] = arrays['intrinsic']
            for si, side in enumerate(['left', 'right']):
                names = [side+'Hand'] + [side+'Thumb'+v for v in ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip']]
                for finger in ['Index', 'Middle', 'Ring', 'Little']:
                    names += [side+finger+'Finger'+v for v in ['Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip']]
                for j, name in enumerate(names):
                    transform = np.repeat(np.eye(4)[None], n, 0)
                    transform[:, :3, 3] = arrays['hand_points'][:, si, j] @ c
                    f['transforms/'+name] = transform
                    f['confidences/'+name] = arrays['hand_confidence'][:, si, j]
        source_row['source'] = str(hdf)
        if supplied is None:
            record['body_adapter'] = 'egodex'
    with resource_lock('user-recording-registry'):
        registry = s.uploads()
        registry[ident] = record
        s.write(s.ROOT / 'user-data/registry.json', registry)
    return {k: v for k, v in public.items() if k != 'owner'}


def attach_dataset_routes(app, lock):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='dataset-io')
    for path in (s.ROOT/'dataset-exports').glob('*/state.json'):
        if s.read(path,{}).get('status') in {'queued','running'}:
            s.write(path,{'status':'failed','error':'Export interrupted by a service restart; request a fresh export'})

    @app.middleware('http')
    async def same_origin(request, call_next):
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('origin')
            trusted_proxy=request.client and request.client.host in {'127.0.0.1','::1'} and origin in {'http://localhost:8765','http://127.0.0.1:8765'}
            if origin and not trusted_proxy and origin != str(request.base_url).rstrip('/') and origin.split('://', 1)[-1] != request.headers.get('host'):
                return JSONResponse({'detail': 'Use the application origin'}, status_code=403)
        return await call_next(request)

    @app.get('/library')
    def library(request: Request, response: Response):
        token = request.cookies.get(s.COOKIE)
        if not token:
            token = secrets.token_urlsafe(32)
            response.set_cookie(s.COOKIE, token, httponly=True, secure=request.url.scheme == 'https' or request.headers.get('x-forwarded-proto') == 'https', samesite='strict', max_age=60*60*24*180, path='/')
        response.headers['Cache-Control']='private, no-store'
        public = s.read(s.STORE / 'manipulation4d/catalog.json', [])
        own = [r['public'] for r in s.uploads().values() if r['public']['owner'] == s.owner(token)]
        return [{k: v for k, v in row.items() if k != 'owner'} for row in [*public, *own]]

    @app.get('/uploads/template')
    def template():
        # Instructions and an executable packer, not fabricated motion data.
        return FileResponse(Path(__file__).with_name('pack_recording.py'), media_type='text/x-python', filename='pack_arenalabs_recording.py')

    @app.post('/uploads')
    async def upload(request: Request, file: UploadFile = File(...)):
        token = request.cookies.get(s.COOKIE)
        if not token:
            raise HTTPException(409, 'Open Data library to initialize your workspace')
        if not (file.filename or '').lower().endswith('.zip'):
            raise HTTPException(400, 'Upload the ZIP created with the recording template')
        ident = 'upload-' + uuid.uuid4().hex
        directory = s.ROOT / 'user-data' / ident
        directory.mkdir(parents=True)
        archive = directory / 'original.zip'
        size = 0
        try:
            with archive.open('wb') as output:
                while block := await file.read(1024*1024):
                    size += len(block)
                    if size > MAX_ARCHIVE:
                        raise ValueError('Compressed upload limit is 2 GiB')
                    output.write(block)
            # Keep validation off the event loop while streaming other requests.
            import asyncio
            return await asyncio.get_running_loop().run_in_executor(pool, import_recording, archive, ident, s.owner(token))
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            s.write(directory / 'import-error.json', {'error': str(error)[:300]})
            raise HTTPException(400, str(error)[:300]) from error
        finally:
            await file.close()

    def run_export(ident, record):
        directory = s.ROOT / 'dataset-exports' / ident
        state = directory / 'state.json'
        s.write(state, dict(status='running', phase='Packaging synchronized modalities'))
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = paths.pythonpath()
            python = os.environ.get('ARENA_DATASET_PYTHON', str(s.ROOT / 'venv/bin/python'))
            with (directory / 'worker.log').open('w') as log:
                subprocess.run([python, '-m', 'arenalab_workbench.dataset_io.worker', str(directory / 'request.json')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = s.read(directory / 'result.json')
            s.write(state, {**result, 'status': 'completed', 'download_url': f'/api/4d/exports/{ident}/download'})
        except Exception:
            s.write(state, dict(status='failed', error='Dataset packaging failed while validating motion or assets. Server log retained; retry after the error is resolved.'))

    @app.post('/exports')
    def start_export(body: ExportRequest, request: Request):
        require_scene(body.scene_id, request)
        from arenalab_workbench.height_registry import public_entry, default_id
        sid = default_id(body.scene_id) or body.scene_id
        require_scene(sid, request)
        public = public_entry(sid)
        if not public:raise HTTPException(409, 'Recording is not connected')
        from .export import collect_raw_sources, collect_video_sources
        scene_path,scene = select_export_scene(body,sid,public)
        for job in getattr(app.state,'generation_jobs',{}).values():
            if job.get('scene_id')==sid and job.get('robot_id')==body.robot_id:
                current=s.read(s.ROOT/'queue'/job['id']/'state.json',{})
                if current.get('status') in {'queued','running'}:raise HTTPException(409,'Generation is still running; download after the remaining modalities finish')
        ident = uuid.uuid4().hex
        directory = s.ROOT / 'dataset-exports' / ident
        record = dict(id=ident, owner=s.owner(request.cookies.get(s.COOKIE)), scene_id=sid, robot_id=body.robot_id, scene_path=str(scene_path), raw_sources=collect_raw_sources(sid,scene,body.robot_id), video_sources=collect_video_sources(sid,scene))
        s.write(directory / 'request.json', record)
        s.write(directory / 'state.json', {'status': 'queued'})
        pool.submit(run_export, ident, record)
        return {'id': ident, 'status': 'queued'}

    def export_record(ident, request):
        if len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise HTTPException(404, 'Unknown export')
        directory = s.ROOT / 'dataset-exports' / ident
        record = s.read(directory / 'request.json', {})
        if record.get('owner') != s.owner(request.cookies.get(s.COOKIE)) or not record:
            raise HTTPException(404, 'Unknown export')
        return directory

    @app.get('/exports/{ident}')
    def export_status(ident: str, request: Request):
        return {'id': ident, **{k:v for k,v in s.read(export_record(ident, request) / 'state.json').items() if k!='path'}}

    @app.get('/exports/{ident}/download')
    def download(ident: str, request: Request):
        directory = export_record(ident, request)
        result = s.read(directory / 'state.json')
        if result['status'] != 'completed':
            raise HTTPException(409, 'Export is not ready')
        return FileResponse(result['path'], media_type='application/zip', filename='arenalabs-lerobot-' + ident[:8] + '.zip')


def protect_uploaded_assets(app):
    @app.middleware('http')
    async def private_assets(request, call_next):
        private=False
        if request.url.path.startswith('/assets/'):
            try:
                path = s.within(s.STORE, request.url.path.removeprefix('/assets/'))
                private=any((parent/'.workspace.json').exists() for parent in path.parents if parent.is_relative_to(s.STORE.resolve()))
                if path.name == '.workspace.json' or not s.permitted(path, request.cookies.get(s.COOKIE)):
                    return JSONResponse({'detail': 'Asset unavailable'}, status_code=404)
            except ValueError:
                return JSONResponse({'detail': 'Asset unavailable'}, status_code=404)
        response=await call_next(request)
        if private:response.headers['Cache-Control']='private, no-store'
        return response
