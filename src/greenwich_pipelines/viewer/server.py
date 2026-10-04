"""Loopback-only Motion Lab service. Explicit catalog paths, no cloud work."""
import argparse,json,os,subprocess,sys,threading,uuid,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .export import array_json,robot_model,motion_payload,join_playback_segments
from .uploads import unpack_robot,MAX_BYTES
from .sources import prepare_source

PAGES={'locomotion':'fullbody','localmanipulation':'nymeria','manipulation':'gento'}

class Conversion(BaseModel):
    clip_id:str
    robot_id:str

def create_app(config_path):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding='utf-8'))
    root=config_path.parent.parent;storage=Path(config['storage']).resolve();storage.mkdir(parents=True,exist_ok=True)
    robots={r['id']:r for r in config['robots']};clips={c['id']:c for c in config['clips']}
    jobs={};lock=threading.Lock();video_lock=threading.Lock();pool=ThreadPoolExecutor(max_workers=1);app=FastAPI(title='arenalab Motion Lab')
    app.state.executor=pool
    def public_robot(row):return {k:v for k,v in row.items() if k in {'id','label','kind','pages','model_url','neutral_url','status','report','conversion_ready'}}
    def public_clip(row):
        result={k:v for k,v in row.items() if k in {'id','page','title','family','description','frames','fps','duration','source_name','source_format','preview','cache','version','metadata'}}
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
        with lock:return dict(page=page,robots=[public_robot(r) for r in robots.values() if page in r['pages']],clips=[public_clip(c) for c in clips.values() if c['page']==page])
    def convert_task(jobid,clip,robot):
        job=jobs[jobid]
        try:
            job['status']='running';folder=storage/'jobs'/jobid;folder.mkdir(parents=True)
            roster=folder/'robots.json';array_json(roster,[robot['pipeline_row']])
            spec=dict(schema='greenwich.job.v1',id='viewer_'+jobid,pipeline=PAGES[clip['page']],source=str(prepare_source(clip,folder)),robots=str(roster),runtime=config['runtime'],output=str(folder/'output'),fps=30)
            if 'max_seconds' in clip:spec['max_seconds']=clip['max_seconds']
            array_json(folder/'job.json',spec)
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-m','greenwich_pipelines','run',str(folder/'job.json')],stdout=log,stderr=subprocess.STDOUT,cwd=root)
            if result.returncode:raise RuntimeError('Conversion failed. Logs and intermediate results are retained in the local job directory.')
            report=json.loads((folder/'output/result.json').read_text(encoding='utf-8'))
            motions=[a for a in report['artifacts'] if a['kind']=='motion']
            motions.sort(key=lambda a:a.get('clip',''))
            payload=join_playback_segments(motion_payload(robot['pipeline_row'],folder/'output'/a['path'],PAGES[clip['page']]) for a in motions)
            payload.update(clip_id=clip['id'],robot_id=robot['id'],version='Current pipeline · Motion quality not evaluated')
            export=folder/'playback.json';array_json(export,payload);url=f'/assets/jobs/{jobid}/playback.json'
            with lock:clip.setdefault('cache',{})[robot['id']]=url;job.update(status='completed',url=url)
        except Exception as exc:job.update(status='failed',error=str(exc))
    @app.post('/api/convert')
    def convert(request:Conversion):
        clip=clips.get(request.clip_id);robot=robots.get(request.robot_id)
        if not clip or not robot or clip['page'] not in robot['pages']:raise HTTPException(400,'Invalid clip / robot pair')
        cached=clip.get('cache',{}).get(robot['id'])
        if cached:return dict(status='completed',url=cached,cached=True)
        if not robot.get('conversion_ready'):raise HTTPException(409,robot.get('report',{}).get('message','Mechanical metadata required. Model preview only.'))
        with lock:
            existing=next((j for j in jobs.values() if j['clip_id']==clip['id'] and j['robot_id']==robot['id'] and j['status'] in {'queued','running'}),None)
            if existing:return existing.copy()
            jid=uuid.uuid4().hex[:16];job=dict(id=jid,status='queued',clip_id=clip['id'],robot_id=robot['id']);jobs[jid]=job
        pool.submit(convert_task,jid,clip,robot)
        return job.copy()
    @app.get('/api/jobs/{job_id}')
    def get_job(job_id:str):
        if job_id not in jobs:raise HTTPException(404,'Unknown job')
        return jobs[job_id].copy()
    @app.post('/api/robots/{page}')
    async def upload(page:str,file:UploadFile=File(...)):
        if page not in PAGES:raise HTTPException(404,'Unknown pipeline')
        content=await file.read(MAX_BYTES+1);rid='custom_'+uuid.uuid4().hex[:10];folder=storage/'uploaded'/rid
        try:
            xml,label,report=unpack_robot(content,file.filename or '',folder)
            row=dict(id=rid,label=label,kind='Custom URDF',pages=[page],status=report['status'],report=report,conversion_ready=False)
            if not report['missing_meshes']:
                try:
                    robot_model(dict(xml=str(xml)),folder)
                    row.update(model_url=f'/assets/uploaded/{rid}/model.glb',neutral_url=f'/assets/uploaded/{rid}/neutral.json')
                except Exception as exc:report.update(status='invalid_model',message=f'Model loading failed: {exc}');row['status']='invalid_model'
            with lock:robots[rid]=row;array_json(storage/'uploaded/registry.json',[r for r in robots.values() if r['id'].startswith('custom_')])
            return public_robot(row)
        except Exception as exc:raise HTTPException(400,str(exc)) from exc
    registry=storage/'uploaded/registry.json'
    if registry.exists():
        for row in json.loads(registry.read_text(encoding='utf-8')):robots[row['id']]=row
    app.mount('/assets',StaticFiles(directory=storage),name='assets')
    dist=root/'web/dist'
    if dist.exists():
        app.mount('/web',StaticFiles(directory=dist),name='web')
        @app.get('/')
        @app.get('/{page}')
        def index(page='locomotion'):
            if page not in PAGES:raise HTTPException(404,'Unknown page')
            return FileResponse(dist/'index.html')
    return app

def main():
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/viewer.local.json'));a=p.parse_args()
    uvicorn.run(create_app(a.config),host='127.0.0.1',port=8765)

if __name__=='__main__':main()
