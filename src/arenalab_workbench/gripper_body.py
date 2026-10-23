"""Head + two gripper observations -> native EgoAllo and SMPL-H stages."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,sys,subprocess,json,hashlib
import numpy as np
from arenalab_workbench.gripper_source import ROOT,save,digest
from arenalab_workbench.egoallo_runtime import gpu_environment
from arenalab_workbench.human_reference import family_for_work
POLICY='gripper-head-two-ee-smplh-v2'
def body_path(work):
 work=Path(work);meta=json.loads((work/'source.json').read_text());cal=meta.get('head_calibration')
 if meta.get('motion_source'):
  from arenalab_workbench.source_identity import absolute_spec,robot_source_identity
  spec=absolute_spec(meta['motion_source'],work);raw=Path(spec['path'])
  identity=dict(source=digest(work/'source.npz'),spec=spec,raw=digest(raw),adapter='supplied-gripper-body-v1')
  if spec.get('format')=='robot_joint_state':identity['complete_robot']=robot_source_identity(spec,dict(source=str(work/'source.npz'),frames=meta['frames'],fps=meta.get('fps',30)))
  if spec.get('objects_path'):
   objects=Path(spec['objects_path']);objects=objects if objects.is_absolute() else work/objects
   identity['objects']=digest(objects)
  return work/('body-supplied-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24])
 suffix=hashlib.sha256(json.dumps(cal,sort_keys=True).encode()).hexdigest()[:12] if cal else None
 return work/('body-smplh-v2-camera-'+suffix if suffix else 'body-smplh-v2')

def prepare(work):
 work=Path(work);body=body_path(work);body.mkdir(exist_ok=True)
 src=np.load(work/'source.npz');meta=json.loads((work/'source.json').read_text());camera=src['camera'];points=src['tcp_zup'];n=len(points);headR=camera[:,:3,:3]@np.diag([-1.,-1.,1.]);headp=camera[:,:3,3].copy()
 supplied=meta.get('motion_source');calibration=meta.get('head_calibration');audit=None
 if supplied:calibration=None # Complete supplied body bypasses the head-to-body reconstruction.
 if calibration:
  if calibration['reference']!='lower_neck' or calibration['policy']!='neck-camera-cpf-v1':raise ValueError('Unsupported declared camera reference')
  # The source camera is immutable. Only derive the model input reference.
  model=paths.egoallo/'model.npz'
  with np.load(model,allow_pickle=True) as smpl:
   vertices=smpl['v_template'];regressor=smpl['J_regressor']
   if regressor.shape==():regressor=regressor.item()
   joints=regressor@vertices
   pupil=(vertices[6260]+vertices[6262]+vertices[2800]+vertices[2802])/4
   neck_to_cpf=pupil-joints[12] # Standard SMPL-H joint 12: neck (root included).
  offset=np.asarray(calibration['neck_to_camera_optical_m'],float)
  neck=headp-np.einsum('tij,j->ti',camera[:,:3,:3],offset)
  headp=neck+np.einsum('tij,j->ti',headR,neck_to_cpf)
  audit=dict(**calibration,neutral_neck_to_cpf_m=neck_to_cpf.tolist(),model_sha256=digest(model),orientation='Camera orientation retained as approximate head orientation')
  np.savez_compressed(body/'head-reference-audit.npz',source_camera=camera,derived_neck_position=neck,derived_cpf_position=headp)
 times=src['timestamps_s'] if 'timestamps_s' in src else np.arange(n)/30
 np.savez_compressed(body/'observations.npz',head_position=np.r_[headp[:1],headp],head_rotation=np.concatenate([headR[:1],headR]),wrist_position=points,palm_position=points,palm_forward=src['tool_R_zup'][:,:,:,2],palm_normal=src['tool_R_zup'][:,:,:,0],timestamps_s=times,floor_z=src['floor_z'])
 save(body/'input_manifest.json',dict(clip=work.name,frames=n,fps=30,dataset=meta['dataset'],policy=POLICY,body_inputs=['camera/head pose','left gripper reference pose','right gripper reference pose'],excluded=['recorded whole-body joints','recorded pelvis pose as prior','recorded feet','recorded human SMPL'],source_sha256=digest(work/'source.npz'),source_trajectory_edited=False,height_calibration=meta.get('height_calibration'),reference=meta['target_reference'],head_calibration=calibration,head_reference_audit=audit,head_proxy='Declared lower-neck camera mount plus neutral SMPL-H neck-to-CPF anatomy' if calibration else 'Source camera/device pose; no invented pupil translation'))
 (body/'operation-accuracy').mkdir(exist_ok=True);np.savez_compressed(body/'operation-accuracy/source-targets.npz',points=points[:,:,None])
 if supplied:
  manifest=json.loads((body/'input_manifest.json').read_text())
  spec=dict(supplied)
  for key in ('path','objects_path'):
   if spec.get(key) and not Path(spec[key]).is_absolute():spec[key]=str((work/spec[key]).resolve())
  manifest.update(motion_source=spec,policy='supplied-gripper-body-v1',end_effector='robotiq-2f-140',body_inputs=['supplied complete '+spec['family'].upper()+' motion'],excluded=[],head_proxy='Recorded camera retained as an observation; not used to reconstruct the supplied body')
  save(body/'input_manifest.json',manifest)
  return body
 base=paths.egoallo
 for name in ['model.npz','egoallo_checkpoint_april13']:
  if not (body/name).exists():(body/name).symlink_to(base/name)
 return body

def run(work):
 work=Path(work);body=body_path(work)
 supplied=json.loads((work/'source.json').read_text()).get('motion_source')
 if body.exists() and not supplied:family_for_work(body)
 if (body/'prior-ready.json').exists():
  report=json.loads((body/'prior-ready.json').read_text());assert report['source_sha256']==digest(work/'source.npz');assert all(digest(body/k)==v for k,v in report['raw_sha256'].items());return body
 body=prepare(work)
 if supplied:
  from arenalab_workbench.motion_backend import normalized_source
  normalized=normalized_source(body)
  save(body/'prior-ready.json',dict(policy='supplied-gripper-body-v1',body_route='supplied_body',human_model=supplied['family'],normalized_source=str(normalized),source_sha256=digest(work/'source.npz'),raw_sha256={k:digest(body/k) for k in ['observations.npz','input_manifest.json','operation-accuracy/source-targets.npz']},source_trajectory_edited=False))
  return body
 from arenalab_workbench.body_inference import run as infer
 infer(body)
 save(body/'prior-ready.json',dict(policy=POLICY,body_window_policy='egoallo-window-300-overlap-60-v1-candidate',source_sha256=digest(work/'source.npz'),raw_sha256={k:digest(body/k) for k in ['observations.npz','egoallo_smplh.npz','smplh_motion.npz','canonical.npz']},source_trajectory_edited=False));return body
if __name__=='__main__':print(run(sys.argv[1]),flush=True)
