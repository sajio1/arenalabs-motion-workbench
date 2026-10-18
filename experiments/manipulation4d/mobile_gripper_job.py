"""Reuse gripper scene/body/compositor with native mobile retargeting."""
from pathlib import Path
import os,sys,json,time
from concurrent.futures import ThreadPoolExecutor
from gpu_scheduler import run as gpu_run
from gripper_job import sources,POLICY
from gripper_body import body_path
from scene_job import save,depth_current
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d')
def run(scene_id,rid,hand,status):
 row=sources()[scene_id];work=Path(row['work']);dest=Path(row['destination']);code=Path(__file__).parent;start=time.time()
 dest.mkdir(parents=True,exist_ok=True)
 scene=json.loads((dest/'scene.json').read_text()) if (dest/'scene.json').exists() else None
 def stage(script,args,python=sys.executable,leased=True):
  phase='Mobile '+script.removesuffix('.py').replace('_',' ')
  save(status,dict(status='running',phase=phase,started_at=start))
  with (work/(rid+'-'+script+'.log')).open('w') as log:
   import subprocess
   (gpu_run if leased else subprocess.run)([python,str(code/script),*map(str,args)],env=os.environ.copy(),stdout=log,stderr=__import__('subprocess').STDOUT,check=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[pool.submit(stage,'gripper_body.py',[work])]
  if not scene or scene.get('geometry',{}).get('visual_policy')!='gripper-visual-v4' or not depth_current(scene):jobs.append(pool.submit(stage,'gripper_scene.py',[work,dest],str(ROOT/'venv/bin/python'),False))
  for j in jobs:j.result()
 stage('wheeled_job.py',[rid,body_path(work),dest,hand,'--gripper-work',work])
 stage('gripper_render.py',[work,rid,dest],str(ROOT/'venv/bin/python'))
 scene=json.loads((dest/'scene.json').read_text());scene['generation_status']=dict(status='completed',policy=POLICY,robot_id=rid,hand_id=hand);save(dest/'scene.json',scene)
 url='/assets/'+(dest/'scene.json').relative_to('/opt/arenalab/store').as_posix()
 save(status,dict(status='completed',phase='Mobile robot, gripper and scene ready',scene_url=url,robot_id=rid,robot_ready=True,elapsed_seconds=time.time()-start));return url
