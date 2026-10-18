"""Validate closed full assets, estimated transforms and unchanged source robot targets."""
from pathlib import Path
import json,hashlib,numpy as np,trimesh
ROOT=Path('/home/ubuntu/arenalab-experiments/manipulation4d');STORE=Path('/opt/arenalab/store');DEST=STORE/'manipulation4d/reconstruction-v2';candidate=json.loads((DEST/'scene-complete-v5-candidate.json').read_text());old=json.loads((DEST/'scene.json').read_text());report={}
for field in ['robots','source_hands_url','source_camera','video','tactile']:
 assert candidate[field]==old[field],field
report['unchanged']=['robot motion and palm targets','source hands','source cameras','source video','video contact annotations']
for row in candidate['completed_assets']:
 full=trimesh.load(STORE/row['full_asset_url'].removeprefix('/assets/'),force='mesh',process=False);browser=trimesh.load(STORE/row['model_url'].removeprefix('/assets/'),force='mesh',process=False);assert full.is_watertight and np.isfinite(full.vertices).all() and np.isfinite(browser.vertices).all()
 frames=json.loads((STORE/row['track_url'].removeprefix('/assets/')).read_text());assert len(frames)==133;fixed_scales={};n=0
 for i,f in enumerate(frames):
  assert f['source_frame']==2*i
  for instance in f['instances']:
   a=np.asarray(instance['matrix']);assert a.shape==(4,4) and np.isfinite(a).all() and np.allclose(a[3],[0,0,0,1]);scale=np.cbrt(np.linalg.det(a[:3,:3]));assert scale>0;rot=a[:3,:3]/scale;assert np.allclose(rot.T@rot,np.eye(3),atol=1e-5)
   previous=fixed_scales.setdefault(instance['id'],scale);assert abs(previous-scale)<1e-6
   bounds=np.asarray(instance['bounds']);assert bounds.shape==(2,3) and np.isfinite(bounds).all() and (bounds[1]>=bounds[0]).all();n+=1
 report[row['kind']]=dict(full_mesh_watertight=True,full_faces=len(full.faces),browser_faces=len(browser.faces),frames=133,estimated_instance_poses=n,segmentation_ids=len(fixed_scales),scale_fixed_per_instance=True,hidden_surface_accuracy_measured=False)
assert candidate['completion_default'] and candidate['completion_available'];report['backside']='Generated model estimate; shape/pose/contact ground truth unavailable'
(DEST/'complete-assets-checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
