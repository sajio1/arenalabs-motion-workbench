"""Camera-conditioned depth and the existing Qwen/SAM scene stages."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,json,sys,subprocess,shutil
import numpy as np,cv2
from concurrent.futures import ThreadPoolExecutor
from arenalab_workbench.gpu_scheduler import run as gpu_run
from arenalab_workbench.gripper_source import ROOT,C,save,digest

def run(work,dest):
 work=Path(work);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);previous_asset=json.loads((dest/'scene.json').read_text()) if (dest/'scene.json').exists() else {}; src=np.load(work/'source.npz');meta=json.loads((work/'source.json').read_text());url='/assets/'+dest.relative_to(str(paths.store)).as_posix()+'/';n=meta['frames'];videos=dest/'videos';videos.mkdir(exist_ok=True)
 for side,path in [('head',meta['head_video']),*[(v['side'],v['path']) for v in meta['wrist_videos']]]:
  dst=videos/(side+'-clip.mp4')
  if not dst.exists():subprocess.run(['ffmpeg','-v','error','-y','-ss',str(meta['start_seconds']),'-i',path,'-t',str(n/30),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)],check=True)
 public=dict(id=work.name,title=meta['task']['label'] if isinstance(meta['task'],dict) else meta['task'],dataset=meta['dataset'],video=dict(url=url+'videos/head-clip.mp4',start_seconds=0))
 if meta.get('height_calibration'):public['title']+=' · '+str(meta['height_calibration']['minimum_height_cm'])+' cm'
 wrist=[dict(side=v['side'],url=url+'videos/'+v['side']+'-clip.mp4',start_seconds=0) for v in meta['wrist_videos']]
 common=dict(**public,fps=30,frames=n,duration=(n-1)/30,world='Source world',robots=[],objects=[],semantic=dict(task=public['title'],objects=[],provenance='Recording annotations'),tactile=dict(status='unavailable',reason='Gripper recording has no measured tactile force; finger estimates do not apply'),wrist_videos=wrist,geometry=dict(source_trajectory_edited=False,ground=dict(height_m=float(src['floor_z']),estimated=True,provenance=meta['floor_provenance']),source_reference=meta['target_reference'],width_provenance=meta['width_provenance']))
 scene=work/('scene-mapanything-v1' if paths.depth_backend=='mapanything' else ('scene-camera-v3' if 'intrinsic' not in src else 'scene'));scene.mkdir(exist_ok=True);
 if paths.depth_backend=='mapanything':
  previous=next((p for p in [work/'scene-camera-v3',work/'scene'] if (p/'preparation.json').exists()),None)
  if previous:
   for name in ['frames','frames.npz','preparation.json','masks','prompts.json','semantic.json','cameras.npz','camera-calibration.json','hardware-classification.json','hardware-tracking.json','hardware-flow-audit.json']:
    old=previous/name;new=scene/name
    if old.exists() and not new.exists():
     if old.is_dir():shutil.copytree(old,new)
     else:shutil.copy2(old,new)
 if scene.name=='scene-camera-v3' and (work/'scene').exists():
  for name in ['frames','frames.npz','preparation.json','masks','prompts.json','semantic.json','depth.npz','depth.json']:
   previous=work/'scene'/name;target=scene/name
   if previous.exists() and not target.exists() and not (scene/'camera-calibration.json').exists():
    if previous.is_dir():shutil.copytree(previous,target)
    else:shutil.copy2(previous,target)
 row=dict(source=str(work/'source.npz'),frames=n,fps=30,video=dict(path=meta['head_video'],start_seconds=meta['start_seconds']));save(scene/'manifest.json',dict(public=public,source=row,destination=str(dest),url=url,source_sha256=digest(work/'source.npz')))
 code=Path(__file__).parent;env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),PYTHONPATH=paths.pythonpath(paths.sam),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
 def stage(script,args):
  if script=='segment_scene.py':script='segment_parallel.py'
  with (scene/(script+'-'+(Path(args[-1]).name if args else 'run')+'.log')).open('w') as log:(subprocess.run if script in {'export_scene_cloud.py','segment_parallel.py','fill_hardware_mask_gaps.py'} or (args and args[-1]=='prepare') else gpu_run)([str(paths.scene_python),str(code/script),*args],env=env.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
 if not (scene/'preparation.json').exists():stage('scene_inference.py',[str(scene),'prepare'])
 if 'intrinsic' in src:
  camera=src['camera'].copy();camera[:,:3,:3]=C@camera[:,:3,:3];camera[:,:3,3]=camera[:,:3,3]@C.T;image_size=meta.get('camera_image_size',[640,480]);K=np.diag([960/image_size[0],540/image_size[1],1])@src['intrinsic'];np.savez_compressed(scene/'cameras.npz',poses=camera,intrinsic=K)
 elif paths.depth_backend=='mapanything' and not (scene/'cameras.npz').exists():
  camera=src['camera'].copy();camera[:,:3,:3]=C@camera[:,:3,:3];camera[:,:3,3]=camera[:,:3,3]@C.T
  np.savez_compressed(scene/'camera-poses.npz',poses=camera)
 def reconstruct_depth():
  if paths.depth_backend=='mapanything':
   from arenalab_workbench.depth_store import current
   if not current(scene):stage('scene_inference.py',[str(scene),'depth'])
   if 'intrinsic' not in src:
    from arenalab_workbench.gripper_camera import bind
    if bind(work,scene):stage('scene_inference.py',[str(scene),'depth'])
   return
  # All sources use the same DA3 model. Recorded depth is retained only as an
  # audit artifact; an older cache must not silently bypass this policy.
  if (scene/'depth.npz').exists():
   info=json.loads((scene/'depth.json').read_text()) if (scene/'depth.json').exists() else {}
   if info.get('model')!='depth-anything/DA3NESTED-GIANT-LARGE-1.1':
    backup=scene/'pre-da3-depth';backup.mkdir(exist_ok=True)
    for name in ['depth.npz','depth.json']:
     if (scene/name).exists():shutil.move(str(scene/name),str(backup/name))
  if not (scene/'depth.npz').exists():stage('scene_inference.py',[str(scene),'depth'])
  if 'intrinsic' not in src:
   from arenalab_workbench.gripper_camera import bind
   bind(work,scene)
   if not (scene/'depth.npz').exists():stage('scene_inference.py',[str(scene),'depth'])
 branches=ThreadPoolExecutor(max_workers=1)
 depth_future=branches.submit(reconstruct_depth)
 from arenalab_workbench.discovery_output import POLICY as DISCOVERY_POLICY
 if not (scene/'semantic.json').exists() or json.loads((scene/'semantic.json').read_text()).get('policy')!=DISCOVERY_POLICY:
  archive=scene/('pre-semantic-v2-'+str(__import__('time').time_ns()));archive.mkdir()
  for p in [*(scene/'masks').glob('object-*-masks.npz'),*(scene/'masks').glob('human-capture-hardware-masks.npz'),*(scene).glob('hardware-*.json')]:shutil.move(str(p),str(archive/p.name))
  stage('scene_inference.py',[str(scene),'discover'])
 prompts=json.loads((scene/'prompts.json').read_text());prompts=[p for p in prompts if p[0] not in {'human-robot-arm','human-robot-gripper'}]+[['human-robot-arm','robot arm'],['human-robot-gripper','robot gripper'],['human-handheld-gripper','handheld robot gripper'],['human-controller','handheld motion controller'],['human-controller-ring','black tracking ring on handheld controller'],['human-game-controller','game controller'],['human-vr-controller','VR controller'],['human-black-gripper','black gripper']];prompts=list({p[0]:p for p in prompts}.values());save(scene/'prompts.json',prompts)
 env.update(ARENA_SEGMENT_FRAMES=str(scene/'frames'),ARENA_SEGMENT_OUTPUT=str(scene/'masks'),ARENA_SEGMENT_PROMPTS=json.dumps(prompts))
 missing=[p for p in prompts if not (scene/'masks'/f'{p[0]}-masks.npz').exists()]
 if missing:
  env['ARENA_SEGMENT_PROMPTS']=json.dumps(missing);stage('segment_scene.py',[])
 if not (scene/'hardware-classification.json').exists():stage('classify_capture_hardware.py',[str(scene)])
 if not (scene/'hardware-tracking.json').exists():stage('propagate_capture_hardware.py',[])
 if not (scene/'hardware-flow-audit.json').exists():stage('fill_hardware_mask_gaps.py',[str(scene)])
 depth_future.result();branches.shutdown()
 stage('export_scene_cloud.py',[str(scene)]);asset=json.loads((dest/'scene.json').read_text());asset['wrist_videos']=wrist;asset['semantic']['source_task']=common['semantic']['task'];
 if not asset['semantic'].get('task'):asset['semantic'].update(task=common['semantic']['task'],task_provenance='Recording annotation')
 asset['geometry'].update(common['geometry']);asset['geometry']['body_status']='Gripper retargeting';asset['geometry']['visual_policy']='gripper-visual-v4';asset['geometry']['scene_work']=str(scene);asset['geometry']['scene_registration']='source_camera_estimated' if 'intrinsic' not in src else 'recorded';
 if previous_asset.get('geometry',{}).get('source_sha256')==digest(work/'source.npz'):
  for key in ['robots','patch_videos','patch_kinds','patch_depth_policy','patch_unavailable']:
   if key in previous_asset:asset[key]=previous_asset[key]
 save(dest/'scene.json',asset)
if __name__=='__main__':run(*sys.argv[1:])
