"""Complete per-recording generation; raw observations and priors remain immutable."""
from arenalab_workbench.settings import paths
from pathlib import Path
from greenwich_pipelines.input_policy import POLICY as ENCODER_POLICY
import json,os,subprocess,sys,time,hashlib
from arenalab_workbench.scene_job import resolve_clip,run as scene_run,save,depth_current,refresh_scene
from arenalab_workbench.body_prior import folder as body_folder
from concurrent.futures import ThreadPoolExecutor
from arenalab_workbench.gpu_scheduler import run as gpu_run, resource_lock, merge_scene
from arenalab_workbench.progressive_job import Progress, initialize, merge_perception
from arenalab_workbench.support_reference import POLICY as SUPPORT_POLICY
ROOT=Path(os.environ.get('ARENA_WORKBENCH_ROOT',str(paths.workbench)))
STORE=paths.store

def run(scene_id,robot_id,hand_id,status):
 # Drain legacy whole-source jobs before entering the stage-level scheduler.
 with resource_lock("source:"+scene_id):pass
 with resource_lock("robot:"+scene_id+":"+robot_id+":"+hand_id):
  return _run(scene_id,robot_id,hand_id,status)

def _run(scene_id,robot_id,hand_id,status):
 from arenalab_workbench.gripper_job import sources,run as gripper_run
 if scene_id in sources():return gripper_run(scene_id,robot_id,hand_id,status)
 started=time.time();public,row=resolve_clip(scene_id)
 expected=public.get('end_effector','none' if public['dataset']=='Nymeria' else 'wuji')
 if hand_id!=expected:raise ValueError('Source requires '+expected)
 # Run body inference independently from scene perception, using a separate status.
 work=body_folder(scene_id)
 scene_url=public.get('scene_url') or '/assets/manipulation4d/progressive/'+hashlib.sha256(scene_id.encode()).hexdigest()+'/scene.json'
 destination=(STORE/scene_url.removeprefix('/assets/')).parent
 from arenalab_workbench.dataset_io.storage import mark_private
 mark_private(destination,public.get('owner'))
 initialize(destination/'scene.json',public,int(row['frames']))
 progress=Progress(status,scene_url,robot_id)
 progress.write('running','Preparing shared body and scene')
 def recover_body():
  requested=time.time()
  with resource_lock('body:'+str(work)):
   acquired=time.time();cache_hit=(work/'prior-ready.json').exists()
   with (status.parent/'body-worker.log').open('w') as log:
    runner=subprocess.run if (work/'prior-ready.json').exists() else gpu_run
    body_env=os.environ.copy()
    from arenalab_workbench.robot_registry import roster
    if roster()[robot_id]['kind']!='Wheeled':body_env['ARENA_STREAM_ROBOT_ID']=robot_id
    runner([sys.executable,str(Path(__file__).with_name('body_prior.py')),scene_id,'--status',str(status.with_name('body-stage.json'))],env=body_env,stdout=log,stderr=subprocess.STDOUT,check=True)
  save(status.with_name('body-timing.json'),dict(cache_hit=cache_hit,lock_wait_seconds=acquired-requested,seconds=time.time()-acquired))
  return work
 def recover_scene():
  with resource_lock('scene:'+scene_id):
   current=json.loads((destination/'scene.json').read_text())
   source_hash=current.get('geometry',{}).get('source_sha256')
   if source_hash and source_hash!=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest():raise ValueError('Source changed; stale scene cannot be reused')
   if not depth_current(current):
    fresh_url=scene_run(scene_id,status.with_name('scene-stage.json'))
    merge_perception(destination/'scene.json',json.loads((STORE/fresh_url.removeprefix('/assets/')).read_text()))
  return json.loads((destination/'scene.json').read_text())
 branches=ThreadPoolExecutor(max_workers=2)
 body_future=branches.submit(recover_body)
 scene_future=branches.submit(recover_scene)
 work=body_future.result()
 scene=json.loads((destination/'scene.json').read_text())
 from arenalab_workbench.robot_registry import roster
 robot=roster()[robot_id]
 mobile=robot['kind']=='Wheeled'
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=paths.pythonpath(),ARENA_SOURCE_WORK=str(work),ARENA_RETARGET_WORK=str(work/'retarget'/robot_id),ARENA_EXPORT_DIR=str(destination),ARENA_SCENE_WORK=str(ROOT/'scene-jobs'/destination.name))
 # Find the immutable depth/mask work by the scene directory's build key.
 env['ARENA_SCENE_WORK']=str(ROOT/'scene-jobs'/scene.get('geometry',{}).get('scene_work',destination.name))
 if (work/'retarget'/f'{robot_id}-result').exists():env['ARENA_RETARGET_WORK']=str(work/'retarget')
 native_validation=destination/('robot-'+robot_id)/'validation.json'
 if native_validation.exists():
  native_dir=json.loads(native_validation.read_text()).get('native_work_dir')
  if native_dir and Path(native_dir).resolve().is_relative_to(work.resolve()):env['ARENA_RETARGET_WORK']=native_dir
 robot_regenerated=False
 stages=[('retarget_robot.py',[robot_id],'Retargeting body, palms and Wuji fingers'),('render_recording.py',[robot_id],'Rendering robot arms and hands in source FOV'),('contact_recording.py',[],'Estimating individual finger contact from video')]
 if hand_id=='none':stages=[('retarget_robot.py',[robot_id],'Retargeting body and observed wrists; no dexterous hand')]
 if mobile:
  stages=[('wheeled_job.py',[robot_id,str(work),str(destination),hand_id],'Retargeting SMPL-X to native mobile body'),('mobile_render_job.py',[str(work),str(destination),robot_id],'Rendering native robot arms and end effector')]+([] if hand_id=='none' else [('contact_recording.py',[],'Estimating individual finger contact from video')])
 def execute_stage(task):
  script,args,phase=task
  stage_started=time.time()
  validation=destination/('robot-'+robot_id)/'validation.json'
  if script=='retarget_robot.py' and validation.exists():
   report=json.loads(validation.read_text())
   native_ready=(Path(env['ARENA_RETARGET_WORK'])/'operation-accuracy/palm'/robot_id/'motion.npz').exists()
   if native_ready and report.get('encoder_policy')==ENCODER_POLICY and report.get('support_policy')==SUPPORT_POLICY and report.get('source_id')==scene_id and all(hashlib.sha256((work/name).read_bytes()).hexdigest()==value for name,value in report.get('source_hashes',{}).items()) and report.get('source_hashes'):
    save(status.with_name(status.stem+'-'+script+'.json'),dict(status='completed',phase=phase,cache_hit=True,seconds=time.time()-stage_started));return
  nonlocal robot_regenerated
  if script=='retarget_robot.py':robot_regenerated=True
  render_receipt=destination/('robot-'+robot_id)/'render-cache.json'
  render_key=None
  if script=='render_recording.py':
   manifest=json.loads((destination/'scene.json').read_text())
   render_key=hashlib.sha256((destination/('robot-'+robot_id)/'playback.json').read_bytes()+json.dumps(manifest.get('geometry',{}),sort_keys=True).encode()).hexdigest()
  if script=='render_recording.py' and not robot_regenerated and (destination/(robot_id+'-arm-view-v1.mp4')).exists() and render_receipt.exists() and json.loads(render_receipt.read_text()).get('key')==render_key and manifest.get('patch_videos',{}).get(robot_id):
   save(status.with_name(status.stem+'-'+script+'.json'),dict(status='completed',phase=phase,cache_hit=True,seconds=time.time()-stage_started));return
  if script=='contact_recording.py' and (destination/'video-contact.json').exists():return
  save(status.with_name(status.stem+'-'+script+'.json'),dict(status='running',phase=phase,started_at=stage_started))
  with (work/(script+'-'+robot_id+'.log')).open('w') as log:
   gpu_run([sys.executable,str(Path(__file__).with_name(script)),*args],env=env,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
  if script=='render_recording.py':save(render_receipt,dict(key=render_key))
  save(status.with_name(status.stem+'-'+script+'.json'),dict(status='completed',phase=phase,started_at=stage_started,seconds=time.time()-stage_started))
 def visual_contact():
  staging=destination/'contact-stage'
  initialize(staging/'scene.json',public,int(row['frames']))
  contact_env=env.copy();contact_env.update(ARENA_EXPORT_DIR=str(staging),ARENA_DEFER_PROXIMITY='1')
  with resource_lock('contact:'+scene_id):
   for candidate in [destination/'video-contact.json',staging/'video-contact.json']:
    if candidate.exists():
     data=json.loads(candidate.read_text())
     manifest=json.loads((work/'input_manifest.json').read_text())
     if data.get('source_sha256')==manifest['source_sha256'] and data.get('source_hands_sha256')==hashlib.sha256((work/'source_hands.npz').read_bytes()).hexdigest():return candidate
   try:
    with (work/'contact-worker.log').open('w') as log:
     gpu_run([sys.executable,str(Path(__file__).with_name('contact_recording.py'))],env=contact_env,stdout=log,stderr=subprocess.STDOUT,check=True)
   except subprocess.CalledProcessError as error:
    import re
    lines=(work/'contact-worker.log').read_text(errors='replace').splitlines()
    reason=next((line.strip() for line in reversed(lines) if re.match(r'[\w.]+(?:Error|Exception):',line)), 'Contact subprocess exited with code '+str(error.returncode))
    raise RuntimeError('Finger contact estimation failed: '+reason) from error
   return staging/'video-contact.json'
 contact_future=branches.submit(visual_contact) if hand_id=='wuji' else None
 try:
  execute_stage(stages[0])
  if robot_regenerated:merge_scene(destination/'scene.json',lambda scene:scene.get('patch_videos',{}).pop(robot_id,None))
  progress.publish('motion','Motion ready; completing point cloud')
  scene=scene_future.result()
  env['ARENA_SCENE_WORK']=str(ROOT/'scene-jobs'/scene.get('geometry',{}).get('scene_work',destination.name))
  progress.publish('scene','Point cloud ready; completing FOV and contact')
  with ThreadPoolExecutor(max_workers=2) as post:
   futures=[post.submit(execute_stage,task) for task in stages[1:] if task[0]!='contact_recording.py']
   for index,future in enumerate(futures):
    future.result()
    progress.publish('fov' if index==0 else 'contact','FOV ready; completing contact' if index==0 else 'Contact ready')
  if contact_future:
   contact_path=contact_future.result()
   from arenalab_workbench.gpu_scheduler import write
   write(destination/'video-contact.json',json.loads(contact_path.read_text()))
   contact_path=destination/'video-contact.json'
   merge_scene(destination/'scene.json',lambda scene:scene.setdefault('tactile',{}).update(status='video_estimate',url='/assets/'+contact_path.relative_to(STORE).as_posix(),reason='Independent per-digit video estimate; not measured force'))
   # Geometry proximity is CPU work and starts only after the DA3 scene exists.
   subprocess.run([sys.executable,str(Path(__file__).with_name('proximity_recording.py'))],env=env,check=True)
   progress.publish('contact','Contact ready')
  def finish(scene):
   if hand_id=='none':scene['tactile']={'status':'unavailable','reason':'No dexterous hand requested for this recording'}
   scene['generation_status']=dict(status='completed',robot_id=robot_id,hand_id=hand_id,seconds=time.time()-started)
  merge_scene(destination/'scene.json',finish)
  progress.complete()
  return scene_url
 except Exception as error:
  progress.write('failed','Generation stage failed: '+type(error).__name__)
  raise
 finally:branches.shutdown()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('scene_id');p.add_argument('robot_id');p.add_argument('hand_id');p.add_argument('--status',type=Path,required=True);a=p.parse_args();run(a.scene_id,a.robot_id,a.hand_id,a.status)
