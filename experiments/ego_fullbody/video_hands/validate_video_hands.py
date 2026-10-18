import os,json,hashlib
from pathlib import Path
import numpy as np,mujoco as mj
r=Path('/home/ubuntu/arenalab-experiments/egoallo');report={}
for name,root in [('jacket',r),('crouch',r/'crouch')]:
 results={}
 for robot in ['h2','a3']:
  new=np.load(root/f'video-hands/{robot}/{robot}-render.npz');old=np.load(root/f'observed-refinement/{robot}/{robot}-render.npz');k=int(new['hand_ranges'][0,0])
  err=float(np.max(np.abs(new['positions'][:,:k]-old['positions'][:,:k])));re=float(np.max(np.abs(new['rotations'][:,:k]-old['rotations'][:,:k])))
  assert err==0 and re==0
  q=np.load(root/f'video-hands/{robot}/wuji_angles.npz')['angles'];assert np.isfinite(q).all();violation=0
  for si,side in enumerate(['left','right']):
   m=mj.MjModel.from_xml_path(str(root/f'wuji/urdf/{side}.urdf'))
   for finger in range(5):
    for j in range(4):
     jid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'{side}_finger{finger+1}_joint{j+1}');lo,hi=m.jnt_range[jid];v=q[:,si,finger*4+j]
     if m.jnt_limited[jid]:violation=max(violation,float(np.maximum(lo-v,v-hi).max()))
  assert violation<1e-5,violation
  results[robot]=dict(body_mesh_position_max_change_m=err,body_mesh_rotation_max_change=re,finite_angles=True,max_limit_violation_rad=violation,source_motion_sha256=hashlib.sha256((root/f'observed-refinement/{robot}/motion.npz').read_bytes()).hexdigest())
 report[name]=results
(r/'video-hand-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
