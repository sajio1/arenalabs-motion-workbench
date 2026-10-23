"""Per-recording immutable body priors, using native EgoAllo/SMPL-H stages."""
from arenalab_workbench.settings import paths
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np,h5py
from scipy.spatial.transform import Rotation
from arenalab_workbench.scene_job import resolve_clip,policy_key,save
from arenalab_workbench.egoallo_runtime import gpu_environment
from arenalab_workbench.human_reference import requested_family, motion_path, SMPLH_POLICY, family_for_work
ROOT=Path(os.environ.get('ARENA_WORKBENCH_ROOT',str(paths.workbench)))
BASE=paths.egoallo
POLICY='observed-palm-body-v2'

def policy_for(dataset):
 return POLICY+'-'+SMPLH_POLICY

def folder(scene_id):
 from arenalab_workbench.height_registry import variant
 custom=variant(scene_id)
 if custom:return Path(custom['body_work'])
 public,row=resolve_clip(scene_id)
 version=policy_for(public['dataset'])+('-nymeria-sparse-nohand-window-v2' if public['dataset']=='Nymeria' else '')
 # Body identity must not depend on DA3/SAM versions or visual artifacts.
 source_hash=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()
 from arenalab_workbench.dataset_io.storage import entry as uploaded_entry
 supplied=row.get('motion_source') or (uploaded_entry(scene_id) or {}).get('motion_source')
 supplied_hashes={}
 if supplied:
  for name in ['path','objects_path']:
   if supplied.get(name):supplied_hashes[name]=hashlib.sha256(Path(supplied[name]).read_bytes()).hexdigest()
 identity=dict(policy=version,motion_source=supplied,supplied_sha256=supplied_hashes,source_sha256=source_hash,window=row.get('source_window'),frames=row['frames'],fps=row['fps'],dataset=public['dataset'])
 if public.get('uploaded'):identity['private_recording_id']=scene_id
 target=ROOT/'body-jobs'/hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
 if target.exists():return target
 # Reuse existing immutable, matching priors without moving their files.
 for ready_path in ([] if supplied else sorted((ROOT/'body-jobs').glob('*/prior-ready.json'))):
  try:
   ready=json.loads(ready_path.read_text());manifest=json.loads(ready_path.with_name('input_manifest.json').read_text())
   if ready.get('policy')!=policy_for(public['dataset']) or ready.get('scene_id')!=scene_id:continue
   if public['dataset']=='Nymeria' and not (ready_path.parent/'windowed-v1').exists():continue
   if manifest.get('source_sha256')==source_hash and manifest.get('source_window')==row.get('source_window') and manifest.get('frames')==row['frames']:return ready_path.parent
  except (OSError,ValueError):continue
 return target

def prepare(scene_id):
 public,row=resolve_clip(scene_id);work=folder(scene_id);work.mkdir(parents=True,exist_ok=True)
 from arenalab_workbench.dataset_io.storage import entry as uploaded_entry
 upload=uploaded_entry(scene_id)
 if row.get('motion_source'):
  spec=dict(row['motion_source']);source=Path(spec['path']);spec['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
  save(work/'input_manifest.json',dict(dataset=public['dataset'],clip=scene_id,frames=int(row['frames']),fps=30,end_effector=public.get('end_effector','none'),source_sha256=spec['sha256'],motion_source=spec,source_trajectory_edited=False,source_window=row.get('source_window'),video=row.get('video')))
  return work
 if not upload:
  from arenalab_workbench.height_registry import variant
  custom=variant(scene_id)
  upload=uploaded_entry(custom['original_scene_id']) if custom else None
 if upload and upload.get('body_adapter')!='egodex':
  from arenalab_workbench.dataset_io.ingest import prepare_body
  prepare_body(upload,work);return work
 if public['dataset']=='Nymeria':
  from arenalab_workbench.nymeria_body import prepare as prepare_nymeria
  prepare_nymeria(public,row,work,save)
  manifest=json.loads((work/'input_manifest.json').read_text());manifest['human_model']=requested_family(public['dataset']);save(work/'input_manifest.json',manifest)
  required=['model.npz','egoallo_checkpoint_april13']
  for name in required:
   if not (work/name).exists():(work/name).symlink_to(BASE/name,target_is_directory=(BASE/name).is_dir())
  return work
 if public['dataset']!='EgoDex' and not upload:raise ValueError('Head/wrist source adapter is not connected for this recording format')
 n=int(row['frames']);start=int(row.get('source_window',{}).get('start_frame',0));sl=slice(start,start+n)
 C=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
 unit=lambda x:x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
 with h5py.File(row['source']) as f:
  cam=f['transforms/camera'][sl];points=[];confidence=[];forward=[];normal=[]
  for side in ['left','right']:
   names=[side+'Hand']+[side+'Thumb'+x for x in ['Knuckle','IntermediateBase','IntermediateTip','Tip']]
   for finger in ['Index','Middle','Ring','Little']:names += [side+finger+'Finger'+x for x in ['Knuckle','IntermediateBase','IntermediateTip','Tip']]
   original=np.stack([f['transforms/'+k][sl,:3,3] for k in names],1);p=original@C.T
   assert len(p)==n and np.isfinite(p).all() and np.max(abs(p@C-original))<1e-10
   co=np.stack([f['confidences/'+k][sl] for k in names],1)
   points.append(p);confidence.append(co);distal=unit(p[:,9]-p[:,0]);forward.append(distal);normal.append(unit(np.cross(p[:,5]-p[:,17],distal))*(-1 if side=='left' else 1))
 points=np.stack(points,1);palm=points[:,:,[5,9,13,17]].mean(2);tips=points[:,:,[4,8,12,16,20]]
 cpfR=C@Rotation.from_matrix(cam[:,:3,:3]).as_matrix()@np.diag([-1.,-1.,1.]);cpfP=cam[:,:3,3]@C.T
 np.savez_compressed(work/'observations.npz',head_position=np.r_[cpfP[:1],cpfP],head_rotation=np.concatenate([cpfR[:1],cpfR]),wrist_position=points[:,:,0],palm_position=palm,palm_forward=np.stack(forward,1),palm_normal=np.stack(normal,1),timestamps_s=np.arange(n)/30,floor_z=float(upload['manifest']['floor_z_m']) if upload else 0.)
 np.savez_compressed(work/'source_hands.npz',points=points,confidence=np.stack(confidence,1),source_to_zup=C)
 (work/'operation-accuracy').mkdir(exist_ok=True)
 np.savez_compressed(work/'operation-accuracy/source-targets.npz',points=np.concatenate([palm[:,:,None],tips,tips[:,:,:2].mean(2)[:,:,None]],2))
 save(work/'input_manifest.json',dict(dataset=public['dataset'],end_effector=public.get('end_effector','wuji'),clip=scene_id,source=row['source'],source_window=row.get('source_window'),frames=n,fps=30,video=row['video'],source_sha256=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest(),source_trajectory_edited=False,cpf_policy='Source optical camera as uncalibrated pupil proxy; no translation extrinsic',floor_policy='EgoDex ARKit source Y=0, declared basis conversion only',fingers='Observed EgoDex 21 landmarks per hand'))
 for name in ['model.npz','egoallo_checkpoint_april13','wuji']:
  if not (work/name).exists():(work/name).symlink_to(BASE/name,target_is_directory=(BASE/name).is_dir())
 return work

def run(scene_id,status=None):
 work=folder(scene_id)
 if work.exists() and (work/'input_manifest.json').is_file() and not json.loads((work/'input_manifest.json').read_text()).get('motion_source'):family_for_work(work)
 if (work/'prior-ready.json').exists():
  ready=json.loads((work/'prior-ready.json').read_text())
  assert all(hashlib.sha256((work/name).read_bytes()).hexdigest()==value for name,value in ready['raw_sha256'].items()),'Immutable body prior changed'
  return work
 work=prepare(scene_id)
 manifest=json.loads((work/'input_manifest.json').read_text())
 if manifest.get('motion_source'):
  from arenalab_workbench.motion_backend import normalized_source
  normalized=normalized_source(work)
  robot_source=manifest['motion_source'].get('format')=='robot_joint_state'
  save(work/'prior-ready.json',dict(scene_id=scene_id,policy='supplied-robot-fullbody-v1' if robot_source else 'supplied-body-direct-v1',body_route='supplied_robot' if robot_source else 'supplied_body',generated_human_model=None,normalized_source=str(normalized),raw_sha256={'input_manifest.json':hashlib.sha256((work/'input_manifest.json').read_bytes()).hexdigest()},source_trajectory_edited=False))
  return work
 from arenalab_workbench.motion_sources import validate_three_point_observations
 validate_three_point_observations(work)
 from arenalab_workbench.height_registry import variant
 custom=variant(scene_id)
 if custom:
  with np.load(work/'observations.npz') as data:observations={k:data[k].copy() for k in data.files}
  observations['floor_z']=np.array(custom['calibration']['floor_z_m']);np.savez_compressed(work/'observations.npz',**observations)
  manifest=json.loads((work/'input_manifest.json').read_text());manifest['height_calibration']=custom['calibration'];save(work/'input_manifest.json',manifest)
 from arenalab_workbench.body_inference import run as infer
 infer(work,status)
 names=['canonical.npz','egoallo_smplh.npz',motion_path(work).name,'observations.npz']
 if (work/'human_reference.json').exists():names.append('human_reference.json')
 if (work/'source_hands.npz').exists():names.append('source_hands.npz')
 save(work/'prior-ready.json',dict(scene_id=scene_id,policy=policy_for(json.loads((work/'input_manifest.json').read_text())['dataset']),body_window_policy='300-frame-60-overlap-committed-frame-stream',raw_sha256={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in names},source_trajectory_edited=False))
 return work

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('scene_id');p.add_argument('--status',type=Path);a=p.parse_args();print(run(a.scene_id,a.status),flush=True)
