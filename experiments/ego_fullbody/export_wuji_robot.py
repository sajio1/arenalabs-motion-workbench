"""Attach Wuji meshes by fixed wrist transforms; retain original native body playback."""
import os
os.environ['WUJI_TELEMETRY_DISABLE']='1'
from pathlib import Path
import json,sys,zipfile
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation as R
from wuji_sdk import RetargetSession,HandModel,Handedness
from greenwich_umi_proof.interactive_viewer import _sample_greenwich_head
from alphamotion.viz.kinematics import visual_mesh_geom_ids

root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent));label=sys.argv[1]
output=Path(os.environ.get('ARENA_OUTPUTDIR',root));output.mkdir(parents=True,exist_ok=True)
row=json.loads((root/(label+'-robot.json')).read_text())[0]
human=np.load(root/'smplx_motion.npz');joints=human['joints'];n=len(joints)
# Standard SMPL-X joints followed by vertex-selected fingertips.
# Use mesh fingertip landmarks directly to avoid optional output joint order changes.
tips={'left':[5361,4933,5058,5169,5286],'right':[8079,7669,7794,7905,8022]}
hand_base={'left':[37,25,28,34,31],'right':[52,40,43,49,46]}
angles=[]
for side,handed,wi in [('left',Handedness.Left,20),('right',Handedness.Right,21)]:
 pts=[joints[:,wi]]
 for base,tip in zip(hand_base[side],tips[side]):pts.extend([joints[:,base],joints[:,base+1],joints[:,base+2],human['vertices'][:,tip]])
 pts=np.stack(pts,1);session=RetargetSession.for_hand(HandModel.WujiHand,side=handed)
 angles.append(np.stack([np.asarray(session.step(np.asarray(p-p[0],dtype=np.float32))) for p in pts]))
angles=np.stack(angles,1);np.savez_compressed(output/'wuji_angles.npz',angles=angles,fps=30,provenance='EgoAllo inferred hand prior, not observed fingers')
if not (root/'wuji').exists():zipfile.ZipFile(root/'wuji-body.zip').extractall(root/'wuji')
motion=Path(os.environ['ARENA_MOTION_PATH']) if 'ARENA_MOTION_PATH' in os.environ else next((root/(label+'-result')).rglob('motion.npz'))
pos,quat,meshes,*_= _sample_greenwich_head(motion,row['xml'],row['body'],rotation_key='rot6d',world_position_key='world_position_cm',floor_y_cm=0.)
rot=R.from_quat(quat.reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(n,-1,3,3)
model=mj.MjModel.from_xml_path(row['xml']);md=mj.MjData(model);mj.mj_forward(model,md);ids=list(visual_mesh_geom_ids(model))
allpos=[];allrot=[];allmesh=[];mounts=[]
keep=[]
for k,g in enumerate(ids):
 owner=model.body(int(model.geom_bodyid[g])).name.lower()
 if 'hand' not in owner and 'finger' not in owner:keep.append(k)
for k in keep:allpos.append(pos[:,k]);allrot.append(rot[:,k]);allmesh.append(meshes[k])
for si,side in enumerate(['left','right']):
 wrist=row['endpoints'][side+'_wrist'];bid=model.body(wrist).id
 gid=next(g for g in ids if model.geom_bodyid[g]==bid);k=ids.index(gid)
 local=R.from_quat(model.geom_quat[gid][[1,2,3,0]]).as_matrix()
 wr=rot[:,k]@local.T;wp=pos[:,k]-np.einsum('tij,j->ti',wr,model.geom_pos[gid])
 # H2 matches the already-reviewed fixed flange. A3 metadata uses its native hand mesh proximal extent.
 if label=='h2':origin=np.array([.058,0,0.])
 else:
  hg=next(g for g in ids if model.body(int(model.geom_bodyid[g])).name.lower()==side+'_hand_link')
  hk=ids.index(hg);verts=meshes[hk][0]
  local_v=(verts@md.geom_xmat[hg].reshape(3,3).T+md.geom_xpos[hg]-md.xpos[bid])@md.xmat[bid].reshape(3,3)
  origin=np.array([float(local_v[:,0].min()),0,0.])
 mount=R.from_euler('y',np.pi/2).as_matrix();hp=wp+np.einsum('tij,j->ti',wr,origin);hr=wr@mount
 mounts.append(dict(side=side,parent=wrist,origin_m=origin.tolist(),rotation=mount.tolist(),status='visual flange integration; not hardware validated'))
 hand=mj.MjModel.from_xml_path(str(root/'wuji/urdf'/(side+'.urdf')));hd=mj.MjData(hand)
 hids=list(visual_mesh_geom_ids(hand));hpos=[];hrot=[]
 for f in range(n):
  for finger in range(5):
   for joint in range(4):
    name=f'{side}_finger{finger+1}_joint{joint+1}'
    jid=mj.mj_name2id(hand,mj.mjtObj.mjOBJ_JOINT,name)
    if jid<0:raise ValueError(name)
    hd.qpos[hand.jnt_qposadr[jid]]=angles[f,si,finger*4+joint]
  mj.mj_forward(hand,hd)
  hpos.append(hp[f]+hd.geom_xpos[hids]@hr[f].T)
  hrot.append(hr[f]@hd.geom_xmat[hids].reshape(-1,3,3))
 hpos=np.asarray(hpos);hrot=np.asarray(hrot)
 for k,g in enumerate(hids):
  mid=hand.geom_dataid[g];a=hand.mesh_vertadr[mid];b=hand.mesh_vertnum[mid];c=hand.mesh_faceadr[mid];e=hand.mesh_facenum[mid]
  allmesh.append((hand.mesh_vert[a:a+b].copy(),hand.mesh_face[c:c+e].copy(),np.array([.60,.72,.64,1.])))
  allpos.append(hpos[:,k]);allrot.append(hrot[:,k])
payload=dict(positions=np.stack(allpos,1),rotations=np.stack(allrot,1),frames=n)
for k,(v,f,c) in enumerate(allmesh):payload[f'v{k}']=v;payload[f'f{k}']=f;payload[f'c{k}']=c
np.savez_compressed(output/(label+'-render.npz'),**payload)
(output/(label+'-wuji-mounts.json')).write_text(json.dumps(mounts,indent=2));print('EXPORTED',label,n,len(allmesh),mounts)
