"""Loopback-only Motion Lab service. Explicit catalog paths, no cloud work."""
import argparse,json,os,subprocess,sys,threading,uuid,hashlib
from concurrent.futures import ThreadPoolExecutor
from queue import SimpleQueue
from pathlib import Path
from typing import Literal
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field
from .export import array_json,robot_model,motion_payload,join_playback_segments
from .uploads import unpack_robot,MAX_BYTES
from .ingest import ingest_humanoid
from .mobile_ingest import ingest_registered_mobile
from .robot_deletion import delete_robot_assets

PAGES={'locomotion':'fullbody','localmanipulation':'nymeria','manipulation':'gento'}

def source_content_identity(clip):
    """Hash declared source dependencies without importing a body model or adapter."""
    from arenalab_workbench.motion_backend import cached_sha
    supplied=clip.get('motion_source') or (clip.get('manifest') or {}).get('motion_source') or {}
    pending=[Path(value) for value in (clip.get('source'),supplied.get('path'),supplied.get('objects_path')) if value]
    result={}
    while pending:
        path=pending.pop().resolve()
        if str(path) in result:continue
        result[str(path)]=cached_sha(path) if path.is_file() else None
        if not path.is_file() or path.suffix.lower()!='.json':continue
        value=json.loads(path.read_text())
        if not isinstance(value,dict):continue
        for row in value.get('sources',[]):
            for key in ('path','options_file'):
                if row.get(key):pending.append(path.parent/row[key])
            if (row.get('options') or {}).get('bind_path'):pending.append(path.parent/row['options']['bind_path'])
        # A source options file may refer to a separate neutral skeleton.
        if value.get('bind_path'):pending.append(path.parent/value['bind_path'])
    return result

def conversion_signature(clip,robot,trajectory_scale='preserve',runtime=None):
    from arenalab_workbench.motion_backend import backend_fingerprint,cached_sha,robot_asset_identity
    environment=os.environ.copy()
    if runtime:environment['ARENA_MOTION_RUNTIME']=str(runtime)
    model=robot.get('pipeline_row',{}).get('xml')
    module=Path(__file__).parents[2]/'arenalab_workbench/unified_conversion.py'
    identity=dict(code=backend_fingerprint(environment),adapter_sha256=cached_sha(module) if module.is_file() else None,
                  pipeline='unified-gpu-hybrid',page=clip['page'],trajectory_scale=trajectory_scale,
                  source=clip,source_sha256=source_content_identity(clip),robot=robot.get('pipeline_row'),
                  model_identity=robot_asset_identity(model) if model else None,fps=30)
    identity['source']={k:v for k,v in clip.items() if k not in {'cache','preview','video'}}
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

def motion_preflight(clip,robot):
    from arenalab_workbench.source_requirements import motion_preflight as check
    return check(clip,robot,source_work=None)

def run_conversion(command,**kwargs):
    from arenalab_workbench.gpu_scheduler import run
    return run(command,label='unified motion conversion',**kwargs)

class Conversion(BaseModel):
    model_config=ConfigDict(extra='forbid')
    clip_id:str
    robot_id:str
    trajectory_scale:Literal['preserve','match_robot']|None=None
    output_modalities:list[Literal['trajectory']]|None=Field(default=None,min_length=1,max_length=1)

class PhysicalMetadata(BaseModel):
    endpoints:dict[str,str]
    palm_frames:dict[str,dict[str,list[float]]]

def create_app(config_path):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding='utf-8'))
    root=config_path.parent.parent;storage=Path(config['storage']).resolve();storage.mkdir(parents=True,exist_ok=True)
    robots={r['id']:r for r in config['robots']};clips={c['id']:c for c in config['clips']}
    # Only signed results from this code/policy may populate the active cache.
    for clip in clips.values():
        clip['cache']={}
    devices=config.get('conversion_devices',[None])
    if not isinstance(devices,list) or not devices or len(set(map(str,devices)))!=len(devices) or (None in devices and len(devices)>1) or any(d is not None and (not isinstance(d,str) or not d.isdecimal()) for d in devices):
        raise ValueError('conversion_devices must be a nonempty list of unique GPU index strings')
    available_devices=SimpleQueue()
    for device in devices:available_devices.put(device)
    jobs={};lock=threading.Lock();video_lock=threading.Lock();pool=ThreadPoolExecutor(max_workers=len(devices));app=FastAPI(title='arenalab Motion Lab')
    conversion_registry=storage/'viewer-conversions.json'
    saved_conversions=json.loads(conversion_registry.read_text()) if conversion_registry.exists() else {}
    mode_cache={}
    app.state.executor=pool
    def source_requirements(scene_id,request,robot_id=None,workflow='locomotion'):
        clip=clips.get(scene_id);robot=robots.get(robot_id)
        if not clip or not robot or clip['page'] not in robot['pages']:raise HTTPException(404,'Clip or robot unavailable')
        return motion_preflight(clip,robot)
    app.state.motion_preflight=source_requirements
    def job_view(job):
        result=job.copy()
        if job['status']=='queued':
            waiting=[j['id'] for j in jobs.values() if j['status']=='queued']
            result['queue_position']=waiting.index(job['id'])+1
        return result
    def public_robot(row):return {k:v for k,v in row.items() if k in {'id','label','kind','pages','model_url','neutral_url','status','report','conversion_ready'}}
    def public_clip(row):
        result={k:v for k,v in row.items() if k in {'id','page','title','family','description','frames','fps','duration','source_name','source_format','preview','cache','version','metadata','has_supplied_body','motion_source_policy','source_body_family','source_evidence','end_effector'}}
        if row.get('video'):
            result['video']={k:v for k,v in row['video'].items() if k not in {'path','browser_transcode'}}
            result['video']['url']=f"/api/source-video/{row['id']}"
        return result
    @app.get('/api/source-video/{clip_id}')
    def source_video(clip_id:str):
        clip=clips.get(clip_id)
        if not clip or not clip.get('video'):raise HTTPException(404,'Source video unavailable')
        path=Path(clip['video']['path'])
        if not path.is_file():raise HTTPException(404,'Source video unavailable')
        if clip['video'].get('browser_transcode'):
            # Source MP4s can contain mp4v, which Chromium cannot decode. Preserve
            # frame cadence and timestamps; cache a browser-compatible H.264 copy.
            import imageio_ffmpeg
            key=hashlib.sha256(f'{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}'.encode()).hexdigest()[:24]
            destination=storage/'source-video'/f'{key}.mp4'
            with video_lock:
                if not destination.exists():
                    destination.parent.mkdir(exist_ok=True)
                    partial=destination.with_suffix('.partial.mp4')
                    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(path),'-map','0:v:0','-an','-vf','scale=960:-2','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-fps_mode','passthrough','-movflags','+faststart',str(partial)],capture_output=True)
                    if result.returncode:raise HTTPException(500,'Source video encoding failed')
                    partial.replace(destination)
            path=destination
        return FileResponse(path,media_type='video/mp4')
    @app.middleware('http')
    async def local_only(request,call_next):
        host=request.headers.get('host','').split(':')[0]
        if host not in {'127.0.0.1','localhost','[','testserver'}:raise HTTPException(403,'Local service only')
        origin=request.headers.get('origin')
        if origin and origin not in {'http://127.0.0.1:8765','http://localhost:8765'}:raise HTTPException(403,'Cross-origin request denied')
        return await call_next(request)
    @app.get('/api/catalog/{page}')
    def catalog(page:str):
        if page not in PAGES:raise HTTPException(404,'Unknown pipeline')
        with lock:return dict(page=page,robots=[public_robot(r) for r in robots.values() if page in r['pages'] and r['id'] not in config.get('hidden_robot_ids',[])],clips=[public_clip(c) for c in clips.values() if c['page']==page],presets=config.get('presets',{}).get(page,[]))
    def convert_task(jobid,clip,robot):
        job=jobs[jobid]
        device=available_devices.get()
        try:
            with lock:job.update(status='running',device=device)
            environment=os.environ.copy()
            if device is not None:environment.update(CUDA_VISIBLE_DEVICES=device,ARENA_GPU_POOL=device)
            environment.setdefault('OMP_NUM_THREADS','4')
            environment.setdefault('MKL_NUM_THREADS','4')
            folder=storage/'jobs'/jobid;folder.mkdir(parents=True)
            environment['ARENA_TRAJECTORY_SCALE']=job['trajectory_scale']
            environment['ARENA_MOTION_BACKEND']='gpu_hybrid'
            spec=dict(clip=clip,robot=robot,trajectory_scale=job['trajectory_scale'],output_modalities=['trajectory'],runtime=config['runtime'],output=str(folder/'output'))
            array_json(folder/'request.json',spec)
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                result=run_conversion([sys.executable,'-m','arenalab_workbench.unified_conversion','--request',str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT,cwd=root,env=environment)
            if result.returncode:
                lines=(folder/'run.log').read_text(encoding='utf-8',errors='replace').splitlines()
                reason=next((line for line in reversed(lines) if line.startswith(('ValueError:', 'RuntimeError:', 'FileNotFoundError:', 'torch.OutOfMemoryError:')) and '/' not in line and '\\' not in line),'Conversion worker exited unexpectedly')
                raise RuntimeError(f'{reason[:240]} · Task {jobid}')
            report=json.loads((folder/'output/report.json').read_text(encoding='utf-8'))
            payload=json.loads((folder/'output/playback.json').read_text(encoding='utf-8'))
            if not payload.get('frames') or payload.get('fps')!=30:raise ValueError('Unified worker returned invalid playback clock')
            payload.update(trajectory_scale=job['trajectory_scale'],clip_id=clip['id'],robot_id=robot['id'],version='Unified GPU hybrid · Kinematic reference')
            export=folder/'playback.json';array_json(export,payload);url=f'/assets/jobs/{jobid}/playback.json'
            with lock:
                if job['trajectory_scale']=='preserve':clip.setdefault('cache',{})[robot['id']]=url
                mode_cache[(clip['id'],robot['id'],job['trajectory_scale'])]=dict(url=url,signature=job['signature'])
                job.update(status='completed',url=url,completed_outputs=['trajectory'],**{k:report[k] for k in ('source_policy','source_policy_version','motion_policy_version','solver_status','solver_converged','quality_status') if k in report})
                saved_conversions[f"{clip['id']}:{robot['id']}:{job['trajectory_scale']}"]=dict(clip_id=clip['id'],robot_id=robot['id'],trajectory_scale=job['trajectory_scale'],signature=job['signature'],url=url)
                array_json(conversion_registry,saved_conversions)
        except Exception as exc:
            with lock:job.update(status='failed',error=str(exc))
        finally:
            available_devices.put(device)
    @app.post('/api/convert')
    def convert(request:Conversion):
        clip=clips.get(request.clip_id);robot=robots.get(request.robot_id)
        if not clip or not robot or clip['page'] not in robot['pages']:raise HTTPException(400,'Invalid clip / robot pair')
        if not robot.get('conversion_ready'):raise HTTPException(409,robot.get('report',{}).get('message','Mechanical metadata required. Model preview only.'))
        requirements=motion_preflight(clip,robot) or {};morphology=requirements.get('morphology') or {}
        if request.trajectory_scale is None and (requirements.get('scale_choice_required') or morphology.get('large_mismatch')):
            raise HTTPException(409,{'code':'SCALE_CHOICE_REQUIRED','message':'Choose preserve or match_robot before conversion.','field':'trajectory_scale'})
        if request.trajectory_scale=='match_robot' and not morphology.get('available'):
            raise HTTPException(409,{'code':'SCALE_ESTIMATE_UNAVAILABLE','message':'Source/robot size estimate is unavailable. Preserve keeps source metres.'})
        mode=request.trajectory_scale or 'preserve'
        signature=conversion_signature(clip,robot,mode,config.get('runtime'))
        cached=mode_cache.get((clip['id'],robot['id'],mode))
        if cached and cached['signature']==signature and (storage/cached['url'].removeprefix('/assets/')).is_file():
            return dict(status='completed',url=cached['url'],cached=True,trajectory_scale=mode,output_modalities=['trajectory'],effective_outputs=['trajectory'],completed_outputs=['trajectory'],**{k:requirements[k] for k in ('source_policy','source_policy_version','motion_policy_version') if k in requirements})
        with lock:
            existing=next((j for j in jobs.values() if j['clip_id']==clip['id'] and j['robot_id']==robot['id'] and j['trajectory_scale']==mode and j['signature']==signature and j['status'] in {'queued','running'}),None)
            if existing:return job_view(existing)
            jid=uuid.uuid4().hex[:16];job=dict(id=jid,status='queued',clip_id=clip['id'],robot_id=robot['id'],trajectory_scale=mode,signature=signature,output_modalities=['trajectory'],effective_outputs=['trajectory'],
                                            **{k:requirements[k] for k in ('source_policy','source_policy_version','motion_policy_version') if k in requirements});jobs[jid]=job
        pool.submit(convert_task,jid,clip,robot)
        with lock:return job_view(job)
    @app.get('/api/jobs/{job_id}')
    def get_job(job_id:str):
        if job_id not in jobs:raise HTTPException(404,'Unknown job')
        with lock:return job_view(jobs[job_id])
    @app.patch('/api/robots/{robot_id}/metadata')
    def update_metadata(robot_id:str,metadata:PhysicalMetadata):
        import numpy as np
        from alphamotion.engine.descriptor import build_from_mjcf
        from greenwich_pipelines.robot_semantics import resolve_endpoints,ROLES
        from greenwich_umi_proof.palm_frames import palm_frame
        robot=robots.get(robot_id)
        if not robot or not robot_id.startswith('custom_') or not robot.get('pipeline_row'):raise HTTPException(400,'A successfully ingested humanoid is required')
        with lock:
            if any(j['robot_id']==robot_id and j['status'] in {'running','queued'} for j in jobs.values()):raise HTTPException(409,'Finish active conversions before changing physical metadata')
            try:
                row=robot['pipeline_row'].copy();spec,*_=build_from_mjcf(row['xml'],row['body'])
                if set(metadata.endpoints)!=set(ROLES):raise ValueError('Exactly four endpoint roles are required')
                resolve_endpoints(spec,metadata.endpoints)
                for role in ROLES[:2]:
                    axes=metadata.palm_frames[role]
                    if np.shape(axes['distal'])!=(3,) or np.shape(axes['normal'])!=(3,):raise ValueError('Each axis must have three components')
                    palm_frame(np.asarray(axes['distal']),np.asarray(axes['normal']))
                row.update(endpoints=metadata.endpoints,palm_frames=metadata.palm_frames)
                folder=storage/'uploaded'/robot_id;descriptor=json.loads((folder/'descriptor.json').read_text());descriptor.update(**row,frame_provenance='User-confirmed physical axes');array_json(folder/'descriptor.json',descriptor)
                robot['pipeline_row']=row;robot['report'].update(endpoints=row['endpoints'],palm_frames=row['palm_frames'],frame_provenance='User-confirmed physical axes')
                for clip in clips.values():clip.get('cache',{}).pop(robot_id,None)
                for key in list(saved_conversions):
                    if saved_conversions[key]['robot_id']==robot_id:saved_conversions.pop(key)
                array_json(conversion_registry,saved_conversions)
                array_json(storage/'uploaded/registry.json',[r for r in robots.values() if r['id'].startswith('custom_')])
            except (ValueError,KeyError) as exc:raise HTTPException(400,str(exc)) from exc
        return public_robot(robot)
    @app.delete('/api/robots/{robot_id}')
    def delete_robot(robot_id:str):
        with lock:
            if any(j['robot_id']==robot_id and j['status'] in {'queued','running'} for j in jobs.values()):
                raise HTTPException(409,'This robot is still generating. Delete it after its current task finishes.')
            try:result=delete_robot_assets(config_path,robot_id)
            except KeyError:raise HTTPException(404,'Robot not found')
            except RuntimeError as exc:raise HTTPException(409,str(exc)) from exc
            except ValueError as exc:raise HTTPException(400,str(exc)) from exc
            for rid in result['deleted_robot_ids']:robots.pop(rid,None)
            config.clear();config.update(json.loads(config_path.read_text()))
            clips.clear();clips.update({c['id']:c for c in config['clips']})
            saved_conversions.clear();saved_conversions.update(json.loads(conversion_registry.read_text()))
        return result
    @app.post('/api/robots/{page}')
    async def upload(page:str,file:UploadFile=File(...)):
        if page not in PAGES:raise HTTPException(404,'Unknown pipeline')
        content=await file.read(MAX_BYTES+1);rid='custom_'+uuid.uuid4().hex[:10];folder=storage/'uploaded'/rid
        try:
            xml,label,report=unpack_robot(content,file.filename or '',folder)
            row=dict(id=rid,label=label,kind='Custom URDF',pages=[page],status=report['status'],report=report,conversion_ready=False)
            if not report['missing_meshes']:
                try:
                    pipeline_row=ingest_registered_mobile(xml,folder,rid,report,list(robots.values())+config.get('robot_templates',[]))
                    mobile=pipeline_row is not None
                    if not mobile and page in {'locomotion','localmanipulation'}:
                        try:pipeline_row=ingest_humanoid(xml,folder,rid,report)
                        except Exception as exc:report.update(status='needs_metadata',message=f'Automatic humanoid ingestion could not complete: {exc}')
                    if pipeline_row:
                        row.update(pipeline_row=pipeline_row,conversion_ready=True,pages=['manipulation'] if mobile else ['locomotion','localmanipulation'],kind='Wheeled' if mobile else 'Humanoid',status='ready')
                    robot_model(pipeline_row or dict(xml=str(xml)),folder)
                    row.update(model_url=f'/assets/uploaded/{rid}/model.glb',neutral_url=f'/assets/uploaded/{rid}/neutral.json')
                except Exception as exc:report.update(status='invalid_model',conversion_ready=False,message=f'Model loading failed: {exc}');row.update(status='invalid_model',conversion_ready=False)
            row['status']=report['status']
            with lock:robots[rid]=row;array_json(storage/'uploaded/registry.json',[r for r in robots.values() if r['id'].startswith('custom_')])
            return public_robot(row)
        except Exception as exc:raise HTTPException(400,str(exc)) from exc
    registry=storage/'uploaded/registry.json'
    if registry.exists():
        for row in json.loads(registry.read_text(encoding='utf-8')):robots[row['id']]=row
    for entry in saved_conversions.values():
        clip=clips.get(entry['clip_id']);robot=robots.get(entry['robot_id'])
        try:
            mode=entry.get('trajectory_scale','preserve')
            if clip and robot and entry['signature']==conversion_signature(clip,robot,mode,config.get('runtime')) and (storage/entry['url'].removeprefix('/assets/')).is_file():
                mode_cache[(clip['id'],robot['id'],mode)]={'url':entry['url'],'signature':entry['signature']}
                if mode=='preserve':clip.setdefault('cache',{})[robot['id']]=entry['url']
        except (OSError,KeyError):pass
    app.mount('/assets',StaticFiles(directory=storage),name='assets')
    dist=root/'web/dist'
    if dist.exists():
        app.mount('/web',StaticFiles(directory=dist),name='web')
        @app.get('/operator/')
        def operator_page():
            page=dist/'operator/index.html'
            if not page.is_file():raise HTTPException(404,'Operator frontend unavailable')
            return FileResponse(page)
        @app.get('/')
        @app.get('/{page}')
        def index(page='locomotion'):
            if page=='operator':return operator_page()
            if page not in PAGES:raise HTTPException(404,'Unknown page')
            return FileResponse(dist/'index.html')
    return app

def main():
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/viewer.local.json'));a=p.parse_args()
    uvicorn.run(create_app(a.config),host='127.0.0.1',port=8765)

if __name__=='__main__':main()
