"""Loopback-only Motion Lab service. Explicit catalog paths, no cloud work."""
import argparse,json,os,subprocess,sys,threading,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .export import array_json,robot_model,motion_payload
from .uploads import unpack_robot,MAX_BYTES

PAGES={'locomotion':'fullbody','localmanipulation':'nymeria','manipulation':'gento'}

class Conversion(BaseModel):
    clip_id:str
    robot_id:str

def create_app(config_path):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding='utf-8'))
    root=config_path.parent.parent;storage=Path(config['storage']).resolve();storage.mkdir(parents=True,exist_ok=True)
    robots={r['id']:r for r in config['robots']};clips={c['id']:c for c in config['clips']}
    jobs={};lock=threading.Lock();pool=ThreadPoolExecutor(max_workers=1);app=FastAPI(title='Greenwich Motion Lab')
    app.state.executor=pool
    def public_robot(row):return {k:v for k,v in row.items() if k in {'id','label','kind','pages','model_url','neutral_url','status','report','conversion_ready'}}
    def public_clip(row):return {k:v for k,v in row.items() if k in {'id','page','title','family','description','frames','fps','duration','source_name','source_format','preview','cache','version'}}
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
            spec=dict(schema='greenwich.job.v1',id='viewer_'+jobid,pipeline=PAGES[clip['page']],source=clip['source'],robots=str(roster),runtime=config['runtime'],output=str(folder/'output'),fps=30)
            if 'max_seconds' in clip:spec['max_seconds']=clip['max_seconds']
            array_json(folder/'job.json',spec)
            with (folder/'run.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-m','greenwich_pipelines','run',str(folder/'job.json')],stdout=log,stderr=subprocess.STDOUT,cwd=root)
            if result.returncode:raise RuntimeError('转换失败。日志保存在本地任务目录；原始和中间结果保留。')
            report=json.loads((folder/'output/result.json').read_text(encoding='utf-8'))
            motions=[a for a in report['artifacts'] if a['kind']=='motion']
            if len(motions)!=1:raise ValueError('Viewer clip must contain one motion segment')
            payload=motion_payload(robot['pipeline_row'],folder/'output'/motions[0]['path'],PAGES[clip['page']])
            payload.update(clip_id=clip['id'],robot_id=robot['id'],version='当前统一管线 · 未做动作质量验收')
            export=folder/'playback.json';array_json(export,payload);url=f'/assets/jobs/{jobid}/playback.json'
            with lock:clip.setdefault('cache',{})[robot['id']]=url;job.update(status='completed',url=url)
        except Exception as exc:job.update(status='failed',error=str(exc))
    @app.post('/api/convert')
    def convert(request:Conversion):
        clip=clips.get(request.clip_id);robot=robots.get(request.robot_id)
        if not clip or not robot or clip['page'] not in robot['pages']:raise HTTPException(400,'Invalid clip / robot pair')
        cached=clip.get('cache',{}).get(robot['id'])
        if cached:return dict(status='completed',url=cached,cached=True)
        if not robot.get('conversion_ready'):raise HTTPException(409,robot.get('report',{}).get('message','机器人尚需机械元数据，当前仅可预览。'))
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
                except Exception as exc:report.update(status='invalid_model',message=f'模型加载失败：{exc}');row['status']='invalid_model'
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
