"""Persistent four-GPU stage generation queue for the production workbench."""
from arenalab_workbench.settings import paths
from pathlib import Path
from greenwich_pipelines.input_policy import POLICY as ENCODER_POLICY
import concurrent.futures,hashlib,json,os,re,subprocess,sys,threading,time,uuid
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel
from arenalab_workbench.hand_uploads import attach_hand_routes
from arenalab_workbench.job_status import job_view
from arenalab_workbench.body_prior import folder as body_folder
from arenalab_workbench.support_reference import POLICY as SUPPORT_POLICY

ROOT=Path(os.environ.get('ARENA_WORKBENCH_ROOT',str(paths.workbench)))

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

from arenalab_workbench.robot_registry import roster, supported

class Generate(BaseModel):
    scene_id:str
    robot_id:str
    hand_id:str='wuji'

class Reconstruct(BaseModel):
    scene_id:str
    robot_id:str='h2'
    hand_id:str='wuji'

def create_api(config_path=None):
    from arenalab_workbench.robot_registry import roster as load_roster
    config_path=Path(config_path or paths.viewer_config)
    def roster():return load_roster(config_path)
    app=FastAPI(title='ArenaLabs manipulation workbench');attach_hand_routes(app,paths.store);folder=ROOT/'queue';folder.mkdir(exist_ok=True)
    from arenalab_workbench.trajectory_height import attach_height_routes
    lock=threading.Lock();pool=concurrent.futures.ThreadPoolExecutor(max_workers=16);jobs={}
    attach_height_routes(app,lock)
    from arenalab_workbench.dataset_io.routes import attach_dataset_routes, require_scene
    attach_dataset_routes(app,lock)
    from arenalab_workbench.scene_geometry import attach_routes
    attach_routes(app)
    for path in folder.glob('*/job.json'):
        job=json.loads(path.read_text());state=path.with_name('state.json')
        if state.exists():
            status=json.loads(state.read_text()).get('status')
            if status in {'queued','running'}:save(state,dict(status='interrupted',phase='Service restarted; retry the conversion'))
        jobs[job['id']]=job
    def view(job):return job_view(job,jobs,folder)
    def execute(job):
        destination=folder/job['id'];state=destination/'state.json'
        save(state,dict(status='running',phase='Starting retargeting'))
        env=os.environ.copy();env.update(ARENA_JOB_ID=job['id'],ARENA_VIEWER_CONFIG=str(config_path),ARENA_GPU_POOL='0,1,2,3',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=paths.pythonpath())
        try:
            with (destination/'worker.log').open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('full_job.py')),job['scene_id'],job['robot_id'],job['hand_id'],'--status',str(state)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                lines=(destination/'worker.log').read_text(errors='replace').splitlines()
                reason=next((line for line in reversed(lines) if re.match(r'[\w.]+(?:Error|Exception):',line)), 'Generation stage failed')
                reason=re.sub(r'/(?:[\w.-]+/)+[\w.-]+','[resource]',reason)
                save(state,{**(json.loads(state.read_text()) if state.exists() else {}),**dict(status='failed',phase='Generation failed',error=reason[:300])});return
            with lock:
                completed=json.loads(state.read_text())
                from arenalab_workbench.trajectory_height import complete as complete_height
                if complete_height(job['scene_id'],completed['scene_url']):return
                from arenalab_workbench.dataset_io import storage as user_storage
                if user_storage.entry(job['scene_id']):
                    from arenalab_workbench.gpu_scheduler import resource_lock
                    with resource_lock('user-recording-registry'):
                        registry=user_storage.uploads();registry[job['scene_id']]['public']['scene_url']=completed['scene_url'];user_storage.write(user_storage.ROOT/'user-data/registry.json',registry)
                    return
                catalog_path=paths.store / 'manipulation4d/catalog.json';catalog=json.loads(catalog_path.read_text())
                for entry in catalog:
                    if entry['id']==job['scene_id']:entry.update(scene_url=completed['scene_url'])
                save(catalog_path,catalog)
        except Exception as error:save(state,{**(json.loads(state.read_text()) if state.exists() else {}),**dict(status='failed',phase='Conversion failed',error=str(error)[:240])})
    @app.get('/robots')
    def robots():
        return [dict(id=r['id'],label=r['label'],kind=r['kind'],model_url=r.get('model_url'),supported=supported(r),experimental=r['kind']=='Wheeled',reason=None if supported(r) else 'Physical end-effector and anatomical metadata is required') for r in roster().values()]
    def enqueue(scene_id,robot_id,hand_id):
        from arenalab_workbench.height_registry import default_id,variant,public_entry
        scene_id=default_id(scene_id) or scene_id
        robot=roster().get(robot_id)
        if not robot:raise HTTPException(400,'Robot is unavailable')
        if not supported(robot):raise HTTPException(409,'Physical end-effector and anatomical metadata is required')
        from arenalab_workbench.gripper_job import sources as gripper_sources,cached as gripper_cached
        gripper_source=scene_id in gripper_sources()
        entry=public_entry(scene_id)
        if not entry:raise HTTPException(400,'Recording is not connected')
        if entry['dataset']=='UMI' and not variant(scene_id):raise HTTPException(409,'Set the first trajectory height before generating UMI')
        expected_hand='robotiq-2f-140' if gripper_source else entry.get('end_effector', 'none' if entry['dataset']=='Nymeria' else 'wuji')
        if hand_id!=expected_hand:raise HTTPException(409,'This source requires '+expected_hand)
        if entry.get('scene_url'):
            scene=json.loads((paths.store/entry['scene_url'].removeprefix('/assets/')).read_text())
            validation_path=(paths.store/entry['scene_url'].removeprefix('/assets/')).parent/('robot-'+robot_id)/'validation.json'
            if gripper_source and gripper_cached(scene_id,robot_id,scene):
                return dict(status='completed',scene_url=entry['scene_url'],robot_id=robot_id,robot_ready=True)
            valid=not gripper_source and any(r['id']==robot_id and r.get('encoder_policy')==ENCODER_POLICY for r in scene.get('robots',[]))
            if robot['kind']=='Wheeled':
                from arenalab_workbench.wheeled_body import POLICY as MOBILE_POLICY
                valid=validation_path.exists() and json.loads(validation_path.read_text()).get('policy')==MOBILE_POLICY
            if validation_path.exists():
                report=json.loads(validation_path.read_text())
                if report.get('source_id'):
                    try:
                        work=body_folder(scene_id);valid=valid and report.get('human_model')=='smplh' and report.get('human_policy')==__import__('arenalab_workbench.human_reference',fromlist=['SMPLH_POLICY']).SMPLH_POLICY and report.get('encoder_policy')==ENCODER_POLICY and report.get('support_policy')==SUPPORT_POLICY and report['source_id']==scene_id and bool(report.get('source_hashes')) and all((work/name).exists() and hashlib.sha256((work/name).read_bytes()).hexdigest()==value for name,value in report['source_hashes'].items())
                    except ValueError:valid=False
            from arenalab_workbench.scene_job import depth_current
            from arenalab_workbench.depth_store import fov_current
            if valid and depth_current(scene) and any(r['id']==robot_id for r in scene['robots']) and fov_current(scene,robot_id) and (hand_id=='none' or scene.get('tactile',{}).get('url')):
                return dict(status='completed',scene_url=entry['scene_url'],robot_id=robot_id,robot_ready=True)
        with lock:
            for job in jobs.values():
                if job.get('kind')=='full' and job['scene_id']==scene_id and job.get('robot_id')==robot_id and job.get('hand_id')==hand_id:
                    current=view(job)
                    if current['status'] in {'queued','running'}:return current
            job=dict(id=uuid.uuid4().hex[:16],kind='full',scene_id=scene_id,robot_id=robot_id,hand_id=hand_id,created_at=time.time());jobs[job['id']]=job
            save(folder/job['id']/'job.json',job);save(folder/job['id']/'state.json',dict(status='queued',phase='Waiting for GPU'))
            pool.submit(execute,job);return view(job)
    @app.post('/reconstruct')
    def reconstruct(body:Reconstruct,request:Request):
        require_scene(body.scene_id,request)
        return enqueue(body.scene_id,body.robot_id,body.hand_id)
    @app.post('/generate')
    def generate(body:Generate,request:Request):
        require_scene(body.scene_id,request)
        return enqueue(body.scene_id,body.robot_id,body.hand_id)
    @app.get('/jobs/{job_id}')
    def status(job_id:str,request:Request):
        with lock:
            if job_id not in jobs:raise HTTPException(404,'Unknown conversion task')
            require_scene(jobs[job_id]['scene_id'],request)
            return view(jobs[job_id])
    app.state.generation_jobs=jobs
    app.state.pool=pool
    return app
