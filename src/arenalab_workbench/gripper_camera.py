"""Source-pose-conditioned depth using learned, explicitly estimated intrinsics."""
from pathlib import Path
import json
import numpy as np
from arenalab_workbench.gripper_source import C,save,digest
POLICY='source-camera-depth-v3'

def bind(work,scene):
 work=Path(work);scene=Path(scene);source=np.load(work/'source.npz');before=digest(work/'source.npz')
 if (scene/'camera-calibration.json').exists():
  audit=json.loads((scene/'camera-calibration.json').read_text())
  if audit['source_sha256']!=before:raise ValueError('Source camera changed; use a new visual workspace')
  return
 original=scene/'depth-camera-prior.npz'
 if not original.exists():(scene/'depth.npz').rename(original);(scene/'depth.json').rename(scene/'depth-camera-prior.json')
 prior=np.load(original);_,h,w=prior['depth'].shape
 K=np.median(prior['intrinsics'],axis=0);K=np.diag([960/w,540/h,1.])@K
 if not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0:raise ValueError('Invalid predicted intrinsic matrix')
 poses=source['camera'].copy();poses[:,:3,:3]=C@poses[:,:3,:3];poses[:,:3,3]=poses[:,:3,3]@C.T
 np.savez_compressed(scene/'cameras.npz',poses=poses,intrinsic=K)
 save(scene/'camera-calibration.json',dict(policy=POLICY,source_sha256=before,source_camera_poses_unchanged=True,intrinsics_estimated=True,estimated=True,intrinsic=K.tolist(),intrinsic_provenance='Median DA3 predicted intrinsics, mapped back to 960x540 inference images',extrinsic_provenance='Recorded head device pose with declared optical-axis basis; optical-center extrinsic not independently calibrated',source_trajectory_edited=False,prior_depth_sha256=digest(original)))
 assert digest(work/'source.npz')==before
