"""Rebuild all published scenes using canonical shared object categories."""
from pathlib import Path
import json,os,sys,subprocess,shutil,time,hashlib
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');CODE=ROOT/'code/manipulation4d';STORE=Path('/opt/arenalab/store');sys.path.insert(0,str(CODE))
from discover_objects import reuse_raw, publish
from discovery_output import normalize_discovery,POLICY,canonical_label
from gpu_scheduler import write,merge_scene
BASE=ROOT/'semantic-v3';BASE.mkdir(exist_ok=True)
def refresh(entry):
 job=BASE/entry['id'];job.mkdir(exist_ok=True);status=job/'status.json'
 publicpath=STORE/entry['scene_url'].removeprefix('/assets/');old=json.loads(publicpath.read_text())
 if status.exists() and json.loads(status.read_text()).get('status')=='completed' and old.get('semantic',{}).get('policy')==POLICY:
  labels=old['semantic']['objects']
  if labels==list(dict.fromkeys(canonical_label(x) for x in labels)):return
 build=hashlib.sha256((CODE/'discovery_output.py').read_bytes()).hexdigest()[:12]
 work=job/('build-'+build);work.mkdir(exist_ok=True)
 try:
  hint=old['geometry']['scene_work'];original=Path(hint) if hint.startswith('/') else ROOT/'scene-jobs'/hint
  assert (original/'depth.npz').exists(),str(original)
  dest=STORE/'manipulation4d/generated'/('semantic-v3-'+entry['id']+'-'+build);dest.mkdir(exist_ok=True)
  manifest=json.loads((original/'manifest.json').read_text());manifest.update(destination=str(dest),url='/assets/'+dest.relative_to(STORE).as_posix()+'/');write(work/'manifest.json',manifest)
  for name in ['frames','frames.npz','preparation.json','depth.npz','depth.json','cameras.npz','camera-intrinsics.npz','camera-calibration.json','lens-calibration.json','discovery-raw.json']:
   src=original/name;target=work/name
   if src.exists() and not target.exists():target.symlink_to(src,target_is_directory=src.is_dir())
  if not reuse_raw(work):
   # Old cache formats still preserve declared proposals; no fabricated labels.
   semantic=json.loads((original/'semantic.json').read_text());value=normalize_discovery(semantic);publish(work,[(0,value)],semantic.get('model','Qwen/Qwen2.5-VL-7B-Instruct'))
  prompts=json.loads((work/'prompts.json').read_text());previous=json.loads((original/'prompts.json').read_text());masks=work/'masks';masks.mkdir(exist_ok=True)
  previous_report=json.loads((original/'masks/segmentation.json').read_text()) if (original/'masks/segmentation.json').exists() else {'prompts':{}}
  report={'model':'SAM 3.1 Object Multiplex','prompts':{}}
  for item in previous:
   if item[0].startswith('human-') and not any(p[0]==item[0] for p in prompts):prompts.append(item)
  reused=[]
  for item in prompts:
   match=next((p for p in previous if p[1]==item[1] and p[0].startswith('human-')==item[0].startswith('human-')),None)
   if match and (original/'masks'/(match[0]+'-masks.npz')).exists():
    shutil.copy2(original/'masks'/(match[0]+'-masks.npz'),masks/(item[0]+'-masks.npz'));reused.append(item[1])
    if match[0] in previous_report.get('prompts',{}):report['prompts'][item[0]]=previous_report['prompts'][match[0]]
  write(masks/'segmentation.json',report);write(work/'prompts.json',prompts)
  missing=[p for p in prompts if not (masks/(p[0]+'-masks.npz')).exists()]
  write(status,dict(status='running',dataset=entry['dataset'],before=old['semantic']['objects'],after=json.loads((work/'semantic.json').read_text())['objects'],new_sam_categories=[p[1] for p in missing],reused=reused,started_at=time.time()))
  env=os.environ.copy();env.update(ARENA_GPU_POOL='0,1,2,3',PYTHONPATH=str(ROOT/'sam3')+':/opt/arenalab/src',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',ARENA_SEGMENT_FRAMES=str(work/'frames'),ARENA_SEGMENT_OUTPUT=str(masks),ARENA_SEGMENT_PROMPTS=json.dumps(missing))
  def stage(name,args):
   with (work/(name+'.log')).open('w') as log:subprocess.run([str(ROOT/'venv/bin/python'),str(CODE/name),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  if missing:stage('segment_parallel.py',[])
  stage('export_scene_cloud.py',[str(work)])
  fresh=json.loads((dest/'scene.json').read_text());fresh['geometry']['scene_work']=str(work)
  def update(scene):
   for k in ['cloud','segmented_instances','objects','semantic']:scene[k]=fresh[k]
   scene.setdefault('geometry',{})['segmentation']=fresh['geometry']['segmentation'];scene['geometry']['scene_work']=str(work)
   # Old proximity is derived from old masks, and is recalculated separately.
   scene.get('tactile',{}).pop('proximity_url',None)
  merge_scene(publicpath,update)
  after=json.loads(publicpath.read_text());assert all(r in after.get('robots',[]) for r in old.get('robots',[])), 'Robot manifest changed concurrently'
  # Recover the existing source hand trajectories for the geometry-only index.
  if old.get('tactile',{}).get('proximity_url'):
   candidates=[ROOT/'recordings'/entry['id'],ROOT/'body-priors'/entry['id']]
   candidates += [p.parent for p in ROOT.rglob('source_hands.npz') if entry['id'] in str(p)]
   source=next((p for p in candidates if (p/'source_hands.npz').exists()),None)
   if source:
    env.update(ARENA_SOURCE_WORK=str(source),ARENA_EXPORT_DIR=str(publicpath.parent));stage('proximity_recording.py',[])
  result=json.loads(status.read_text());result.update(status='completed',policy=POLICY,scene_url=entry['scene_url'],visible_instances=sum(o['visible_frames']>0 for o in fresh['geometry']['segmentation']['objects']),maximum_simultaneous_instances=max(len(f['instances']) for f in fresh['segmented_instances']['frames']),robot_assets_preserved=True,finished_at=time.time());write(status,result);print('COMPLETE',entry['id'],result,flush=True)
 except Exception as exc:
  write(status,dict(status='failed',error=str(exc),dataset=entry['dataset']));print('FAILED',entry['id'],str(exc),flush=True)
catalog=json.loads((STORE/'manipulation4d/catalog.json').read_text());entries=[e for e in catalog if e.get('scene_url')];write(BASE/'inventory.json',entries)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(refresh,entries))
print('ALL_FINISHED',len(entries),flush=True)
