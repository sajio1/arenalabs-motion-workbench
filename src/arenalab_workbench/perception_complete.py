"""Recover explicitly returned VLM category lists, then run normal segmentation."""
from arenalab_workbench.settings import paths
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
from arenalab_workbench.discovery_output import normalize_discovery
ROOT=paths.workbench

def complete_scene():
 source=Path(os.environ['ARENA_SCENE_WORK']);dest=Path(os.environ['ARENA_EXPORT_DIR'])
 from arenalab_workbench.scene_job import depth_current
 current=json.loads((dest/'scene.json').read_text())
 if depth_current(current) and (source/'depth.npz').exists() and (source/'prompts.json').exists():return source
 raw=json.loads((source/'discovery-raw.json').read_text())['raw'];start=next(i for i,c in enumerate(raw) if c in '[{');value=json.JSONDecoder().raw_decode(raw[start:])[0];semantic=normalize_discovery(value)
 scene=json.loads((dest/'scene.json').read_text())
 if semantic['objects']==scene['semantic']['objects']:
  repaired=scene.get('geometry',{}).get('perception_work')
  return ROOT/'perception-jobs'/repaired if repaired and (ROOT/'perception-jobs'/repaired).exists() else source
 key=hashlib.sha256((source.name+json.dumps(semantic,sort_keys=True)).encode()).hexdigest();work=ROOT/'perception-jobs'/key;work.mkdir(parents=True,exist_ok=True)
 output=paths.store / 'manipulation4d/perception'/key;output.mkdir(parents=True,exist_ok=True)
 for name in ['depth.npz','depth.json','frames','frames.npz','preparation.json','cameras.npz']:
  if (source/name).exists() and not (work/name).exists():(work/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
 manifest=json.loads((source/'manifest.json').read_text());manifest.update(destination=str(output),url='/assets/'+output.relative_to(str(paths.store)).as_posix()+'/')
 (work/'manifest.json').write_text(json.dumps(manifest));semantic.update(model=scene['semantic']['model'],provenance='Original VLM category proposals, normalized without manual object labels');(work/'semantic.json').write_text(json.dumps(semantic));prompts=[[f'object-{i+1:02d}',name] for i,name in enumerate(semantic['objects'])]+[['human-hand','hand'],['human-arm','arm']];(work/'prompts.json').write_text(json.dumps(prompts))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),PYTHONPATH=paths.pythonpath(paths.sam),ARENA_SEGMENT_FRAMES=str(work/'frames'),ARENA_SEGMENT_OUTPUT=str(work/'masks'),ARENA_SEGMENT_PROMPTS=json.dumps(prompts))
 if not (output/'scene.json').exists():
  for script,args in [('segment_scene.py',[]),('export_scene_cloud.py',[str(work)])]:
   with (work/(script+'.log')).open('w') as log:subprocess.run([str(paths.scene_python),str(Path(__file__).with_name(script)),*args],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 new=json.loads((output/'scene.json').read_text())
 from arenalab_workbench.gpu_scheduler import merge_scene
 def update(scene):
  for key in ['cloud','segmented_instances','semantic']:scene[key]=new[key]
  scene['geometry']['segmentation']=new['geometry']['segmentation'];scene['geometry']['proposal_parser']='Original VLM categories preserved, no manual category labels';scene['geometry']['perception_work']=work.name
 merge_scene(dest/'scene.json',update)
 return work
if __name__=='__main__':print(complete_scene(),flush=True)
