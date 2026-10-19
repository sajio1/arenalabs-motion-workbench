"""Shared, isolated video -> segmented point-cloud jobs. No source/robot tuning.

All heavy stages are separate processes on dynamically leased GPUs. The manifest, raw
model outputs, masks, source hashes and stage timings remain in the private job.
Scene-only results never borrow robot motion from another recording.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from gpu_scheduler import run as gpu_run

ROOT=Path(os.environ.get('ARENA_WORKBENCH_ROOT','/home/ubuntu/arenalab-experiments/manipulation4d'))
STORE=Path('/opt/arenalab/store/manipulation4d')
POLICY='scene-cloud-v6-da3-sam31'
WORKER_FILES=('scene_inference.py','video_io.py','segment_scene.py','segment_parallel.py','tracking_recovery.py',
              'sam_runtime.py','sam_features.py','export_scene_cloud.py','discovery_output.py',
              'discover_objects.py','nymeria_camera.py','gpu_scheduler.py')
# Bump POLICY when model versions, reconstruction settings or output semantics
# change. Exact worker hashes additionally identify each immutable build artifact.

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    from gpu_scheduler import write
    write(path,value)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def resolve_clip(scene_id):
    from trajectory_height import variant
    custom=variant(scene_id)
    from dataset_io.storage import entry as uploaded_entry
    upload=uploaded_entry(scene_id)
    if upload:return upload['public'],upload['source_row']
    public=custom['public'] if custom else next((r for r in read(STORE/'catalog.json') if r['id']==scene_id),None)
    config=read('/opt/arenalab/configs/viewer.cloud.json')
    row=custom['source_row'] if custom else next((r for r in config['clips'] if r['id']==(public or {}).get('source_catalog_id',scene_id)),None)
    if not public or not row:raise ValueError('Recording is not connected to the source catalog')
    video=Path(row.get('video',{}).get('path',''))
    if not video.is_file():raise ValueError('Source video is missing')
    source=Path(row['source'])
    if not source.is_file():raise ValueError('Source motion data is missing')
    return public,row
def policy_signature(row,source_sha256,video_sha256,policy=POLICY):
    payload=dict(policy=policy,source_sha256=source_sha256,video_sha256=video_sha256,window=row.get('source_window'),frames=row['frames'],fps=row['fps'],video_start=row['video'].get('start_seconds',0),video_frame_times=row['video'].get('frame_times'))
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def policy_key(scene_id):
    _,row=resolve_clip(scene_id)
    return policy_signature(row,digest(row['source']),digest(row['video']['path']))
def cache_key(scene_id):
    public,row=resolve_clip(scene_id)
    h=hashlib.sha256(json.dumps(dict(policy=POLICY,id=scene_id,window=row.get('source_window'),video_start=row.get('video',{}).get('start_seconds',0),frames=row['frames']),sort_keys=True).encode())
    for name in ('scene_job.py', *WORKER_FILES):h.update(Path(__file__).with_name(name).read_bytes())
    for name in [row['source'],row['video']['path']]:h.update(digest(name).encode())
    return h.hexdigest()
def depth_current(scene):
    geometry=scene.get('geometry',{})
    depth=geometry.get('depth',{})
    from discovery_output import POLICY as discovery_policy
    return geometry.get('segmentation',{}).get('policy')=='sam31-da3-semantic-v2' and geometry.get('segmentation',{}).get('discovery_policy') in {'qwen-visible-objects-v2',discovery_policy} and isinstance(depth,dict) and depth.get('model')=='depth-anything/DA3NESTED-GIANT-LARGE-1.1' and (scene.get('dataset')!='Nymeria' or geometry.get('lens_calibration',{}).get('policy')=='aria-fisheye624-upright-pinhole-v1' and geometry.get('camera_calibration',{}).get('policy')=='nymeria-official-handeye-v1')

def refresh_scene(scene_id,status,old_url):
    old_path=Path('/opt/arenalab/store')/old_url.removeprefix('/assets/')
    old=read(old_path);new_url=run(scene_id,status)
    fresh=read(Path('/opt/arenalab/store')/new_url.removeprefix('/assets/'))
    from progressive_job import merge_perception
    merge_perception(old_path,fresh)
    return old_url

def run(scene_id,status):
    key=cache_key(scene_id);work=ROOT/'scene-jobs'/key;work.mkdir(parents=True,exist_ok=True)
    public,row=resolve_clip(scene_id);destination=STORE/'generated'/key;destination.mkdir(parents=True,exist_ok=True)
    from dataset_io.storage import mark_private
    mark_private(destination,public.get('owner'))
    manifest=dict(policy=POLICY,cache_key=key,public=public,source=row,destination=str(destination),url=f'/assets/manipulation4d/generated/{key}/',gpu_pool=[0,1,2,3],source_sha256=digest(row['source']),video_sha256=digest(row['video']['path']),source_trajectory_edited=False)
    code=work/'code';code.mkdir(exist_ok=True)
    for name in WORKER_FILES:shutil.copy2(Path(__file__).with_name(name),code/name)
    manifest['worker_code_sha256']={p.name:digest(p) for p in code.glob('*.py')}
    save(work/'manifest.json',manifest)
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'sam3')+':'+environment.get('PYTHONPATH',''))
    started=time.monotonic();started_at=time.time();timings={};cache_hits={}
    def execute(stage,phase):
        stage_env=environment.copy()
        receipt=work/(stage+'-complete.json')
        if receipt.exists():
            cache_hits[stage]=dict(original_seconds=read(receipt)['seconds']);timings[stage]=0.;return
        save(status.with_name(status.stem+'-'+stage+'.json'),dict(status='running',phase=phase,started_at=started_at,elapsed_seconds=time.monotonic()-started,stage_seconds=timings))
        tic=time.monotonic()
        if stage=='segment':
            prompts=read(work/'prompts.json');stage_env.update(ARENA_SEGMENT_FRAMES=str(work/'frames'),ARENA_SEGMENT_OUTPUT=str(work/'masks'),ARENA_SEGMENT_PROMPTS=json.dumps(prompts))
            stage_env['ARENA_SEGMENT_SCRIPT']=str(code/'segment_scene.py')
            command=[str(ROOT/'venv/bin/python'),str(code/'segment_parallel.py')]
        else:
            script='export_scene_cloud.py' if stage=='export' else 'scene_inference.py'
            command=[str(ROOT/'venv/bin/python'),str(code/script),str(work)]
            if stage!='export':command.append(stage)
        with (work/f'{stage}.log').open('w') as log:result=(gpu_run if stage in {'depth','discover'} else subprocess.run)(command,env=stage_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            lines=(work/f'{stage}.log').read_text(errors='replace').splitlines()
            reason=next((line for line in reversed(lines) if re.match(r'[\w.]+(?:Error|Exception):',line)),f'Worker exited with code {result.returncode}')
            reason=re.sub(r'/(?:[\w.-]+/)+[\w.-]+','[resource]',reason)
            save(status,dict(status='failed',phase=phase+' failed',error=reason[:300],elapsed_seconds=time.monotonic()-started,stage_seconds=timings))
            raise RuntimeError(phase+' failed; raw stage log retained')
        timings[stage]=time.monotonic()-tic
        save(receipt,dict(seconds=timings[stage],worker_code_sha256=manifest['worker_code_sha256']))
        save(status.with_name(status.stem+'-'+stage+'.json'),dict(status='completed',phase=phase,seconds=timings[stage]))
    save(status,dict(status='running',phase='Parallel DA3 depth and object segmentation',started_at=started_at))
    execute('prepare','Preparing source frames')
    with ThreadPoolExecutor(max_workers=2) as branches:
        depth_future=branches.submit(execute,'depth','DA3 depth reconstruction')
        execute('discover','Discovering manipulable objects')
        execute('segment','SAM 3.1 object segmentation')
        depth_future.result()
    execute('export','Exporting segmented point cloud')
    if digest(row['source'])!=manifest['source_sha256'] or digest(row['video']['path'])!=manifest['video_sha256']:raise ValueError('Source changed while reconstructing')
    report=dict(stage_seconds=timings,cache_hits=cache_hits,total_seconds=time.monotonic()-started,gpu_pool=[0,1,2,3],source_sha256=manifest['source_sha256'],video_sha256=manifest['video_sha256'],worker_code_sha256=manifest['worker_code_sha256'],source_trajectory_edited=False,weights_finetuned=False,object_prompts='Qwen visual proposals; shared prompt across recordings',robot_output='Not generated by the scene reconstruction stage')
    scene=read(destination/'scene.json');scene['geometry']['generation']=report;scene['geometry']['scene_work']=key;save(destination/'scene.json',scene);save(destination/'timings.json',report)
    save(status,dict(status='completed',phase='Segmented scene ready',cache_key=key,scene_url=manifest['url']+'scene.json',elapsed_seconds=report['total_seconds'],stage_seconds=timings,robot_ready=False))
    return manifest['url']+'scene.json'
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scene_id');parser.add_argument('--status',type=Path,required=True);args=parser.parse_args();run(args.scene_id,args.status)
