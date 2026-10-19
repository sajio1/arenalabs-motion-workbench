"""Queue entry for head + two EE -> SMPL-X -> robot + gripper generation."""
from arenalab_workbench.settings import paths
from pathlib import Path
from greenwich_pipelines.input_policy import POLICY as ENCODER_POLICY
import json,os,subprocess,sys,time
from arenalab_workbench.gripper_source import ROOT,save,digest
from concurrent.futures import ThreadPoolExecutor
from arenalab_workbench.gpu_scheduler import run as gpu_run, resource_lock, merge_scene
from arenalab_workbench.progressive_job import Progress, initialize, merge_perception
from arenalab_workbench.scene_job import depth_current
from arenalab_workbench.depth_store import fov_current
POLICY='gripper-head-two-ee-smplx-housing-v3'
from arenalab_workbench.gripper_registry import REGISTRY, sources

def cached(scene_id,rid,scene,require_visual=True):
 row=sources().get(scene_id)
 if not row:return False
 from arenalab_workbench.gripper_source import REAL_POLICY
 meta=json.loads((Path(row['work'])/'source.json').read_text())
 if meta['dataset']=='Real robot' and meta.get('adapter_policy')!=REAL_POLICY:return False
 work=Path(row['work']);dest=Path(row['destination']);p=dest/('robot-'+rid)/'validation.json'
 if not p.exists():return False
 report=json.loads(p.read_text());return report.get('encoder_policy')==ENCODER_POLICY and report.get('head_calibration')==json.loads((work/'source.json').read_text()).get('head_calibration') and report.get('policy')==POLICY and report.get('source_sha256')==digest(work/'source.npz') and any(r['id']==rid and r.get('hand')=='robotiq-2f-140' for r in scene['robots']) and (dest/('robot-'+rid)/'model.glb').exists() and (dest/('robot-'+rid)/'playback.json').exists() and (not require_visual or (fov_current(scene,rid) and scene.get('geometry',{}).get('visual_policy')=='gripper-visual-v4' and depth_current(scene)))
def run(scene_id,rid,hand,status):
 if hand!='robotiq-2f-140':raise ValueError('Gripper source requires a mapped gripper end effector')
 row=sources()[scene_id];work=Path(row['work']);dest=Path(row['destination']);start=time.time();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),PYTHONPATH=paths.pythonpath(),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2');code=Path(__file__).parent
 from arenalab_workbench.robot_registry import roster
 robot=roster()[rid]
 mobile=robot['kind']=='Wheeled'
 if not mobile:env['ARENA_STREAM_ROBOT_ID']=rid
 from arenalab_workbench.gripper_body import body_path
 from arenalab_workbench.height_registry import public_entry
 public=public_entry(scene_id)
 if public is None:
  from arenalab_workbench.height_registry import variant
  public=variant(scene_id)['public']
 meta=json.loads((work/'source.json').read_text())
 initialize(dest/'scene.json',public,meta['frames'])
 url='/assets/'+(dest/'scene.json').relative_to(str(paths.store)).as_posix()
 initial=json.loads((dest/'scene.json').read_text())
 all_cached=cached(scene_id,rid,initial,require_visual=True)
 progress=Progress(status,url,rid);progress.write('running','Parallel body and scene generation')
 def recover_body():
  with resource_lock('body:'+str(body_path(work))):
   runner=subprocess.run if (body_path(work)/'prior-ready.json').exists() else gpu_run
   with (work/(rid+'-body.log')).open('w') as log:
    runner([str(paths.scene_python),str(code/'gripper_body.py'),str(work)],env=env.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
 def recover_scene():
  with resource_lock('scene:'+scene_id):
   scene=json.loads((dest/'scene.json').read_text())
   if scene.get('geometry',{}).get('source_sha256')!=digest(work/'source.npz') or scene.get('geometry',{}).get('visual_policy')!='gripper-visual-v4' or not depth_current(scene):
    staging=dest/'perception-stage'
    execute(('gripper_scene.py',[str(work),str(staging)],str(paths.scene_python),'Camera-conditioned depth and SAM scene generation'))
    merge_perception(dest/'scene.json',json.loads((staging/'scene.json').read_text()))
 def execute(task):
  script,args,python,phase=task
  stage_started=time.time()
  save(status.with_name(status.stem+'-'+script+'.json'),dict(status='running',phase=phase,started_at=time.time()))
  with (work/(rid+'-'+script+'.log')).open('w') as log:
   runner=subprocess.run if script in {'gripper_scene.py','gripper_export.py'} else gpu_run
   runner([python,str(code/script),*args],env=env.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
  save(status.with_name(status.stem+'-'+script+'.json'),dict(status='completed',phase=phase,started_at=stage_started,seconds=time.time()-stage_started))
 with ThreadPoolExecutor(max_workers=2) as branches:
  body=branches.submit(recover_body);perception=branches.submit(recover_scene)
  body.result()
  regenerated=not cached(scene_id,rid,json.loads((dest/'scene.json').read_text()),require_visual=False)
  if mobile and not all_cached:
   execute(('wheeled_job.py',[rid,str(body_path(work)),str(dest),hand,'--gripper-work',str(work)],str(paths.scene_python),'Retargeting native mobile body and gripper'))
  elif not mobile and regenerated:
   execute(('gripper_retarget.py',[str(work),rid],str(paths.motion_python),'Retargeting gripper housing centers'))
   execute(('gripper_export.py',[str(work),rid,str(dest)],str(paths.motion_python),'Exporting native robot and gripper'))
  if regenerated:merge_scene(dest/'scene.json',lambda scene:scene.get('patch_videos',{}).pop(rid,None))
  progress.publish('motion','Motion ready; completing point cloud')
  perception.result();progress.publish('scene','Point cloud ready; rendering FOV')
  if not all_cached:execute(('gripper_render.py',[str(work),rid,str(dest)],str(paths.scene_python),'Rendering robot arms and grippers'))
  progress.publish('fov','FOV ready')
 merge_scene(dest/'scene.json',lambda scene:scene.update(generation_status=dict(status='completed',policy=POLICY)))
 progress.complete();return url
