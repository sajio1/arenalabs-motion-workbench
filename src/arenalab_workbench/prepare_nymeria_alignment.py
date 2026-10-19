"""Calibrate Xsens head -> Aria device from synchronized recorded trajectories.

Vectorized implementation of the official Nymeria HandEyeSolver.so3xR3,
stride=2, smooth=False. No robot or action dependent fit and no target edits.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from projectaria_tools.core import data_provider

def solve(A,B):
    a=np.linalg.inv(A[:-2])@A[2:];b=np.linalg.inv(B[:-2])@B[2:]
    av=Rotation.from_matrix(a[:,:3,:3]).as_rotvec().T
    bv=Rotation.from_matrix(b[:,:3,:3]).as_rotvec().T
    u,s,vh=np.linalg.svd(bv@av.T);r=vh.T@u.T
    if np.linalg.det(r)<0:r[2]*=-1
    j=(a[:,:3,:3]-np.eye(3)).reshape(-1,3)
    residual=(b[:,:3,3]@r.T-a[:,:3,3]).reshape(-1)
    t=np.linalg.lstsq(j.T@j,j.T@residual,rcond=None)[0]
    X=np.eye(4);X[:3,:3]=r;X[:3,3]=t
    return X,dict(rotation_singular_values=s.tolist(),translation_condition=float(np.linalg.cond(j.T@j)))

def poses(p,q):
    out=np.repeat(np.eye(4)[None],len(p),0);out[:,:3,:3]=Rotation.from_quat(q).as_matrix();out[:,:3,3]=p;return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('vrs',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
    d=np.load(args.input);vrs=data_provider.create_vrs_data_provider(str(args.vrs))
    query=np.asarray([vrs.convert_from_timecode_to_device_time_ns(int(t*1000))/1000 for t in d['xsens_time_us']])
    times=d['device_time_us'];idx=np.clip(np.searchsorted(times,query),1,len(times)-1)
    idx=np.where(abs(times[idx-1]-query)<abs(times[idx]-query),idx-1,idx)
    valid=(query>times[0]+1e6)&(query<times[-1]-1e6)&(abs(times[idx]-query)<2000)
    A=poses(d['device_position'][idx],d['device_quat_xyzw'][idx]);B=poses(d['head_position'],d['head_quat_wxyz'][:,[1,2,3,0]])
    A=A[valid];B=B[valid]
    if len(A)<1000:raise ValueError('Insufficient synchronized head/device observations')
    X,diagnostics=solve(A,B)
    # Held-out motion consistency, not an absolute scene/contact accuracy claim.
    half=len(A)//2;first,_=solve(A[:half],B[:half]);second,_=solve(A[half:],B[half:])
    a=np.linalg.inv(A[half:-2])@A[half+2:];b=np.linalg.inv(B[half:-2])@B[half+2:]
    err=np.linalg.inv(a@first)@(first@b)
    diagnostics.update(samples=len(A),time_difference_p95_ms=float(np.percentile(abs(times[idx][valid]-query[valid]),95)/1000),heldout_relative_position_p95_mm=float(np.percentile(np.linalg.norm(err[:,:3,3],axis=1),95)*1000),heldout_relative_rotation_p95_deg=float(np.percentile(Rotation.from_matrix(err[:,:3,:3]).magnitude(),95)*180/np.pi),split_translation_difference_mm=float(np.linalg.norm(first[:3,3]-second[:3,3])*1000),split_rotation_difference_deg=float(Rotation.from_matrix(first[:3,:3].T@second[:3,:3]).magnitude()*180/np.pi))
    cal=vrs.get_device_calibration().get_camera_calib('camera-rgb')
    native_from_upright=np.eye(4);native_from_upright[:3,:3]=[[0,1,0],[-1,0,0],[0,0,1]]
    D_C=cal.get_transform_device_camera().to_matrix()@native_from_upright
    H_C=np.linalg.inv(X)@D_C
    np.savez_compressed(args.output,T_device_head=X,T_head_camera=H_C)
    report=dict(policy='nymeria-official-handeye-v1',algorithm='Official Nymeria HandEyeSolver.so3xR3, stride=2, smooth=False',source='https://github.com/facebookresearch/nymeria_dataset/blob/nymeria_dataset_legacy/nymeria/handeye.py',input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),vrs=str(args.vrs),T_device_head=X.tolist(),T_head_camera=H_C.tolist(),diagnostics=diagnostics,source_trajectory_edited=False,extrinsic_provenance='Sequence-level hand-eye calibration from recorded Aria MPS and Xsens head trajectories; RGB optical extrinsic from VRS; upright rotation matches rectification',accuracy_note='Relative motion residuals do not measure absolute scene depth or object contact accuracy')
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
