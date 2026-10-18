"""Immutable gripper-source adapters; no task registration or fitted offsets."""
from pathlib import Path
import json,hashlib,zipfile,os
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path(os.environ.get('ARENA_WORKBENCH_ROOT','/home/ubuntu/arenalab-experiments/manipulation4d'))
C=np.array([[0.,1,0],[0,0,1],[1,0,0]])
# Optical coordinates (right, down, forward) expressed in declared FLU marker axes.
OPTICAL_TO_FLU=np.array([[0.,0,1],[-1,0,0],[0,-1,0]])
GRIPPER_TO_EE=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
REAL_POLICY='recorded-wrist-camera-to-dex1-base-v2'

def recorded_gripper_frames(table,info):
 """Recover actual fixed-base poses from recorded FK camera poses, not teleop goals."""
 if info['recording_metadata'].get('hand_type')!='dex1':
  raise ValueError('Source gripper requires declared fixed base-frame metadata')
 frames=[];provenance=[]
 for side in ['left','right']:
  camera_name=side+'_wrist_cam'
  calibration=info['recording_metadata'].get('camera_calibration',{}).get('observation.images.'+camera_name,{})
  extrinsics=calibration.get('robot_extrinsics')
  if extrinsics is None:
   method=info.get('additional_info',{}).get('repaired',{}).get('method','')
   if info.get('robot_type')!='unitree_g1' or method!='FK g1_dex1_cameras.urdf (verified against vendor pipeline)':raise ValueError('Missing declared camera-to-wrist hardware transform')
   # Same named hardware profile as calibrated G1/DEX1 recordings in this archive.
   transform=np.eye(4);transform[:3,3]=[.055,.003 if side=='left' else -.003,.080]
   extrinsics=dict(source_file='configs/g1_dex1_static_frames.json; shared declared G1/DEX1 hardware profile, not independent calibration',static_frame=dict(parent_body=side+'_wrist_yaw_link',T_child_to_parent=transform.tolist()))
  static=extrinsics['static_frame']
  if static['parent_body']!=side+'_wrist_yaw_link':raise ValueError('Unexpected wrist camera parent')
  transform=np.array(static['T_child_to_parent'],float)
  if not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-6):raise ValueError('Invalid recorded camera rotation')
  recorded=np.asarray(table['observation.camera_extrinsics.'+camera_name+'.cam_to_body'].to_pylist(),float)
  wrist=recorded@np.linalg.inv(transform)
  # Vendor g1_29dof_mode_15_with_dex1_1.urdf: fixed wrist_yaw -> dex1_base.
  base=np.eye(4);base[0,3]=.0415
  frames.append(wrist@base)
  provenance.append(dict(side=side,camera_to_wrist=transform.tolist(),wrist_to_gripper_base=base.tolist(),source=extrinsics['source_file']))
 return np.stack(frames,1),provenance
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def real(source,dest):
 import pyarrow.parquet as pq
 source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 info=json.loads((source/'meta/info.json').read_text());meta=info['recording_metadata'];table=pq.read_table(source/'data/chunk-000/file-000.parquet');a=lambda key:np.array(table[key].to_pylist(),float)
 q=a('observation.state.robot_q_current');n=len(q);rootR=R.from_quat(q[:,[4,5,6,3]]).as_matrix();rootp=q[:,:3];ee=a('observation.state.ee_state').reshape(n,2,6)
 actual,actual_provenance=recorded_gripper_frames(table,info)
 tcp=np.einsum('tij,tsj->tsi',rootR,actual[:,:,:3,3])+rootp[:,None];toolR=rootR[:,None]@actual[:,:,:3,:3]@GRIPPER_TO_EE
 cameras={}
 for name in ['head_cam','left_wrist_cam','right_wrist_cam']:
  c=a('observation.camera_extrinsics.'+name+'.cam_to_body').reshape(n,4,4);out=np.repeat(np.eye(4)[None],n,0);out[:,:3,:3]=rootR@c[:,:3,:3]@OPTICAL_TO_FLU;out[:,:3,3]=np.einsum('tij,tj->ti',rootR,c[:,:3,3])+rootp;cameras[name]=out
 # Raw values are retained; mm is an explicitly labeled schema interpretation,
 # pending a recording-unit declaration. This never changes TCP targets.
 width=a('observation.state.grid_width')/1000
 calibration=meta.get('camera_calibration',{}).get('observation.images.head_cam',{}).get('streams',{}).get('color',{}).get('intrinsics',{}).get('matrix')
 payload=dict(tcp_zup=tcp,tool_R_zup=toolR,width_m=width,raw_grid_width=a('observation.state.grid_width'),floor_z=np.array(0.),camera=cameras['head_cam'],source_ee=ee,timestamps=a('timestamp'))
 if calibration is not None:payload['intrinsic']=np.array(calibration)
 if info.get('fps')!=30:raise ValueError('Only 30 Hz recordings are supported')
 assert np.max(abs(np.einsum('tji,tsj->tsi',rootR,tcp-rootp[:,None])-actual[:,:,:3,3]))<1e-12
 np.savez_compressed(dest/'source.npz',**payload)
 report=dict(dataset='Real robot',frames=n,fps=30,source=str(source),source_sha256=digest(source/'data/chunk-000/file-000.parquet'),source_trajectory_edited=False,body_prior_inputs='Only head/camera and two end-effectors; recorded joints, pelvis and feet excluded',coordinate_conversion='Recorded root transform is used only to express camera and EE observations in their common world; it is not a model target',target_reference='Recorded DEX1 EE marker; marker-to-jaw midpoint hardware validation pending',width_provenance='grid_width interpreted as millimeters from DEX1 range; unit not explicit in schema, not hardware validated',floor_provenance='Recorded IMU-finalized root world Z=0; not a measured floor',camera_provenance='Recorded cam_to_body + recorded root pose + declared FLU-to-optical basis',head_video=str(next((source/'videos').rglob('*head_cam/**/*.mp4'))),start_seconds=0,task=meta['task_name'],wrist_videos=[dict(side=side,path=str(next((source/'videos').rglob('*'+side+'_wrist_cam/**/*.mp4')))) for side in ['left','right']])
 previous_metadata=json.loads((dest/'source.json').read_text()) if (dest/'source.json').exists() else {}
 report.update(adapter_policy=REAL_POLICY,target_reference='Recorded actual DEX1 fixed base frame from wrist-camera FK; vendor fixed wrist-to-base transform',actual_wrist_provenance=actual_provenance,excluded_ee_state='Unspecified teleop/EE frame; fails actual-wrist consistency audit')
 if previous_metadata.get('source_sha256')==report['source_sha256'] and previous_metadata.get('head_calibration'):report['head_calibration']=previous_metadata['head_calibration']
 save(dest/'source.json',report);return report

def umi(source,dest,phase):
 source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(source) as z:
  meta=json.loads(z.read(next(k for k in z.namelist() if k.endswith('.json'))));slug=meta['slug'];n=meta['frames'];start=0 if phase<0 else round(meta['fine'][phase]['s']*30);end=n if phase<0 else round(meta['fine'][phase]['e']*30)
  poses=np.frombuffer(z.read(slug+'_eef.bin'),'<f4').reshape(n,2,7)[start:end].copy();head=np.frombuffer(z.read(slug+'_eefE.bin'),'<f4').reshape(n,7)[start:end].copy();width=np.frombuffer(z.read(slug+'_grip.bin'),'<f4').reshape(n,2)[start:end].astype(float)/1000
  assert meta['pose_up']=='z' and meta['pose_shared'] and meta['fps']==30 and meta['grip_unit']=='mm'
  for name in z.namelist():
   if name.endswith('.mp4'):
    p=dest/Path(name).name
    if not p.exists():p.write_bytes(z.read(name))
 if not np.isfinite(poses).all() or not np.isfinite(head).all():raise ValueError('Missing source poses; no filling or smoothing')
 axes={'+x':[1,0,0],'-x':[-1,0,0],'+y':[0,1,0],'-y':[0,-1,0],'+z':[0,0,1],'-z':[0,0,-1]};bases=[]
 for side in ['L','R']:
  f=np.array(axes[meta['cam_fwd'][side]]);u=np.array(axes[meta['cam_up'][side]]);bases.append(np.stack([u,np.cross(f,u),f],1))
 sr=R.from_quat(poses[:,:,3:].reshape(-1,4)).as_matrix().reshape(end-start,2,3,3);toolR=sr@np.array(bases)[None]
 previous=json.loads(Path('/home/ubuntu/wuji-ego-pilot/estimated-scene.json').read_text());floor=float(np.median(poses[:,:,2]))-previous['source_neutral_waist_above_floor_m'];xy=np.median(head[:,:2],0);forward=np.median(poses[:,:,:2].mean(1),0)-xy;yaw=np.arctan2(forward[1],forward[0])
 f=np.array(axes[meta['cam_fwd']['E']]);u=np.array(axes[meta['cam_up']['E']]);camera=np.repeat(np.eye(4)[None],len(head),0);camera[:,:3,:3]=R.from_quat(head[:,3:]).as_matrix()@np.stack([np.cross(f,u),-u,f],1);camera[:,:3,3]=head[:,:3]
 np.savez_compressed(dest/'source.npz',camera=camera,tcp_zup=poses[:,:,:3].astype(float),tool_R_zup=toolR,width_m=width,raw_poses=poses,raw_head=head,floor_z=np.array(floor),base_xy=xy,base_yaw=np.array(yaw))
 report=dict(dataset='UMI',frames=end-start,fps=30,source=str(source),source_sha256=digest(source),source_trajectory_edited=False,source_frame_range=[start,end],task=({'label':'Complete source recording','s':0,'e':n/30} if phase<0 else meta['fine'][phase]),start_seconds=start/30,metadata=meta,target_reference='Source device-origin proxy; device-to-jaw extrinsic unavailable',width_provenance='Source declared millimeters',floor_provenance='User-authorized source waist-height hypothesis; same estimate for every target; no manual floor offset',estimated_floor_z_m=floor,head_video=str(dest/(slug+'_E.mp4')),wrist_videos=[dict(side=side,path=str(dest/(slug+'_'+view+'.mp4'))) for side,view in [('left','L'),('right','R')]])
 previous_metadata=json.loads((dest/'source.json').read_text()) if (dest/'source.json').exists() else {}
 if previous_metadata.get('source_sha256')==report['source_sha256'] and previous_metadata.get('head_calibration'):report['head_calibration']=previous_metadata['head_calibration']
 save(dest/'source.json',report);return report
if __name__=='__main__':
 import sys
 print(json.dumps(real(sys.argv[2],sys.argv[3]) if sys.argv[1]=='real' else umi(sys.argv[2],sys.argv[3],int(sys.argv[4])),indent=2))
