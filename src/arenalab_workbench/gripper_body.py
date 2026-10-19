"""Head + two gripper observations -> existing EgoAllo and SMPL-X stages."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,sys,subprocess,json,hashlib
import numpy as np
from arenalab_workbench.gripper_source import ROOT,save,digest
from arenalab_workbench.egoallo_runtime import gpu_environment
POLICY='gripper-head-two-ee-smplx-v1'
def body_path(work):
 work=Path(work);meta=json.loads((work/'source.json').read_text());cal=meta.get('head_calibration')
 suffix=hashlib.sha256(json.dumps(cal,sort_keys=True).encode()).hexdigest()[:12] if cal else None
 return work/('body-camera-'+suffix if suffix else 'body')

def prepare(work):
 work=Path(work);body=body_path(work);body.mkdir(exist_ok=True)
 src=np.load(work/'source.npz');meta=json.loads((work/'source.json').read_text());camera=src['camera'];points=src['tcp_zup'];n=len(points);headR=camera[:,:3,:3]@np.diag([-1.,-1.,1.]);headp=camera[:,:3,3].copy()
 calibration=meta.get('head_calibration');audit=None
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
 np.savez_compressed(body/'observations.npz',head_position=np.r_[headp[:1],headp],head_rotation=np.concatenate([headR[:1],headR]),wrist_position=points,palm_position=points,palm_forward=src['tool_R_zup'][:,:,:,2],palm_normal=src['tool_R_zup'][:,:,:,0],timestamps_s=np.arange(n)/30,floor_z=src['floor_z'])
 save(body/'input_manifest.json',dict(clip=work.name,frames=n,fps=30,dataset=meta['dataset'],policy=POLICY,body_inputs=['camera/head pose','left gripper reference pose','right gripper reference pose'],excluded=['recorded whole-body joints','recorded pelvis pose as prior','recorded feet','recorded human SMPL'],source_sha256=digest(work/'source.npz'),source_trajectory_edited=False,height_calibration=meta.get('height_calibration'),reference=meta['target_reference'],head_calibration=calibration,head_reference_audit=audit,head_proxy='Declared lower-neck camera mount plus neutral SMPL-H neck-to-CPF anatomy' if calibration else 'Source camera/device pose; no invented pupil translation'))
 (body/'operation-accuracy').mkdir(exist_ok=True);np.savez_compressed(body/'operation-accuracy/source-targets.npz',points=points[:,:,None])
 base=paths.egoallo
 for name in ['model.npz','SMPLX_NEUTRAL.npz','egoallo_checkpoint_april13']:
  if not (body/name).exists():(body/name).symlink_to(base/name)
 return body

def run(work):
 work=Path(work);body=body_path(work)
 if (body/'prior-ready.json').exists():
  report=json.loads((body/'prior-ready.json').read_text());assert report['source_sha256']==digest(work/'source.npz');assert all(digest(body/k)==v for k,v in report['raw_sha256'].items());return body
 body=prepare(work)
 from arenalab_workbench.body_inference import run as infer
 infer(body)
 save(body/'prior-ready.json',dict(policy=POLICY,body_window_policy='egoallo-window-300-overlap-60-v1-candidate',source_sha256=digest(work/'source.npz'),raw_sha256={k:digest(body/k) for k in ['observations.npz','egoallo_smplh.npz','smplx_motion.npz','canonical.npz']},source_trajectory_edited=False));return body
if __name__=='__main__':print(run(sys.argv[1]),flush=True)
