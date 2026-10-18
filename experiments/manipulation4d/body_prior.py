"""Per-recording immutable body priors, using existing EgoAllo/SMPL-X stages."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np,h5py
from scipy.spatial.transform import Rotation
from scene_job import resolve_clip,policy_key,save
from egoallo_runtime import gpu_environment
ROOT=Path(os.environ.get('ARENA_WORKBENCH_ROOT','/home/ubuntu/arenalab-experiments/manipulation4d'))
BASE=ROOT.parent/'egoallo'
POLICY='observed-palm-body-v2'

def folder(scene_id):
 from trajectory_height import variant
 custom=variant(scene_id)
 if custom:return Path(custom['body_work'])
 public,row=resolve_clip(scene_id)
 version=POLICY+('-nymeria-sparse-nohand-window-v2' if public['dataset']=='Nymeria' else '')
 # Body identity must not depend on DA3/SAM versions or visual artifacts.
 source_hash=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest()
 identity=dict(policy=version,source_sha256=source_hash,window=row.get('source_window'),frames=row['frames'],fps=row['fps'],dataset=public['dataset'])
 if public.get('uploaded'):identity['private_recording_id']=scene_id
 target=ROOT/'body-jobs'/hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
 if target.exists():return target
 # Reuse existing immutable, matching priors without moving their files.
 for ready_path in sorted((ROOT/'body-jobs').glob('*/prior-ready.json')):
  try:
   ready=json.loads(ready_path.read_text());manifest=json.loads(ready_path.with_name('input_manifest.json').read_text())
   if ready.get('policy')!=POLICY or ready.get('scene_id')!=scene_id:continue
   if public['dataset']=='Nymeria' and not (ready_path.parent/'windowed-v1').exists():continue
   if manifest.get('source_sha256')==source_hash and manifest.get('source_window')==row.get('source_window') and manifest.get('frames')==row['frames']:return ready_path.parent
  except (OSError,ValueError):continue
 return target

def prepare(scene_id):
 public,row=resolve_clip(scene_id);work=folder(scene_id);work.mkdir(parents=True,exist_ok=True)
 from dataset_io.storage import entry as uploaded_entry
 upload=uploaded_entry(scene_id)
 if not upload:
  from trajectory_height import variant
  custom=variant(scene_id)
  upload=uploaded_entry(custom['original_scene_id']) if custom else None
 if upload and upload.get('body_adapter')!='egodex':
  from dataset_io.ingest import prepare_body
  prepare_body(upload,work);return work
 if public['dataset']=='Nymeria':
  from nymeria_body import prepare as prepare_nymeria
  prepare_nymeria(public,row,work,save)
  for name in ['model.npz','SMPLX_NEUTRAL.npz','egoallo_checkpoint_april13']:
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
 for name in ['model.npz','SMPLX_NEUTRAL.npz','egoallo_checkpoint_april13','wuji']:
  if not (work/name).exists():(work/name).symlink_to(BASE/name,target_is_directory=(BASE/name).is_dir())
 return work

def run(scene_id,status=None):
 work=folder(scene_id)
 if (work/'prior-ready.json').exists():
  ready=json.loads((work/'prior-ready.json').read_text())
  assert all(hashlib.sha256((work/name).read_bytes()).hexdigest()==value for name,value in ready['raw_sha256'].items()),'Immutable body prior changed'
  return work
 work=prepare(scene_id)
 from trajectory_height import variant
 custom=variant(scene_id)
 if custom:
  with np.load(work/'observations.npz') as data:observations={k:data[k].copy() for k in data.files}
  observations['floor_z']=np.array(custom['calibration']['floor_z_m']);np.savez_compressed(work/'observations.npz',**observations)
  manifest=json.loads((work/'input_manifest.json').read_text());manifest['height_calibration']=custom['calibration'];save(work/'input_manifest.json',manifest)
 if json.loads((work/'input_manifest.json').read_text())['dataset'] in ('Nymeria','EgoDex','Uploaded'):
  from body_inference import run as infer
  infer(work,status)
  names=['canonical.npz','egoallo_smplh.npz','smplx_motion.npz','observations.npz']
  if (work/'source_hands.npz').exists():names.append('source_hands.npz')
  save(work/'prior-ready.json',dict(scene_id=scene_id,policy=POLICY,body_window_policy='300-frame-60-overlap-committed-frame-stream',raw_sha256={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in names},source_trajectory_edited=False))
  return work
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'),EGL_DEVICE_ID=os.environ.get('EGL_DEVICE_ID','3'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH='/opt/arenalab/src',ARENA_EGO_WORKDIR=str(work),XLA_PYTHON_CLIENT_PREALLOCATE='false')
 env=gpu_environment(env)
 for script,phase in [('run_egoallo.py','Reconstructing body prior from head and hands'),('transfer_smplx.py','Transferring body prior to SMPL-X')]:
  if status:save(status,dict(status='running',phase=phase,started_at=time.time()))
  with (work/(script+'.log')).open('w') as log:subprocess.run([str(BASE/'venv/bin/python'),str(Path(__file__).with_name(script))],env=env,cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
 save(work/'prior-ready.json',dict(scene_id=scene_id,policy=POLICY,raw_sha256={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in ['canonical.npz','egoallo_smplh.npz','smplx_motion.npz','source_hands.npz','observations.npz']},source_trajectory_edited=False))
 return work
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('scene_id');p.add_argument('--status',type=Path);a=p.parse_args();print(run(a.scene_id,a.status),flush=True)
