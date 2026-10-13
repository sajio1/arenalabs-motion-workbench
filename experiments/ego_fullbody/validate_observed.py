"""Independent native-motor FK validation of the final file, including mesh feet."""
from pathlib import Path
import os,sys,json
import numpy as np,mujoco as mj
from scipy.spatial.transform import Rotation
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.embodiment.mjcf_build import build_merge
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from greenwich_umi_proof.palm_frames import palm_frame
root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent));label=sys.argv[1]
row=json.loads((root/(label+'-robot.json')).read_text())[0]
model=mj.MjModel.from_xml_path(row['xml']);data=mj.MjData(model)
spec,dof,rest,qnames,_=build_from_mjcf(row['xml'],row['body']);keep,frames=build_merge(model)
roles=['left_wrist','right_wrist','left_ankle','right_ankle'];bids=[model.body(row['endpoints'][s]).id for s in roles]
ids=[list(spec.joint_names).index(row['endpoints'][s]) for s in roles]
C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
obs=np.load(root/'observations.npz');target=obs['wrist_position'];semantic=palm_frame(obs['palm_forward'],obs['palm_normal'])
local=np.stack([palm_frame(row['palm_frames'][s]['distal'],row['palm_frames'][s]['normal']) for s in roles[:2]])
footgeoms=[]
for bid in bids[2:]:
 gids=[]
 for gid in visual_mesh_geom_ids(model):
  b=int(model.geom_bodyid[gid])
  while b>0 and b!=bid:b=int(model.body_parentid[b])
  if b==bid:gids.append(gid)
 footgeoms.append(gids)
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
report={}
for version,path in [('before',next((root/(label+'-result')).rglob('motion.npz'))),('after',root/'observed-refinement'/label/'motion.npz')]:
 if '--before-only' in sys.argv and version=='after':continue
 d=np.load(path);r=d['rot6d'];v1=r[...,:3];v1=v1/np.linalg.norm(v1,axis=-1,keepdims=True);v2=r[...,3:];v2=v2-v1*(v1*v2).sum(-1,keepdims=True);v2/=np.linalg.norm(v2,axis=-1,keepdims=True);rr=np.stack((v1,v2,np.cross(v1,v2)),-1)
 positions=[];rots=[];floors=[];allbounds=[]
 for t in range(len(r)):
  data.qpos[:]=model.qpos0
  for j,names in enumerate(qnames):
   for slot,name in enumerate(names):
    jid=model.joint(name).id;value=d['q'][t,j,slot];data.qpos[model.jnt_qposadr[jid]]=value
    if model.jnt_limited[jid]:allbounds.append(max(model.jnt_range[jid,0]-value,value-model.jnt_range[jid,1],0))
  mj.mj_forward(model,data)
  rootR=C.T@rr[t,0]@C;delta=rootR@data.xmat[frames[keep[0]]].reshape(3,3).T
  rootP=d['root_t'][t]@C/100
  p=rootP+(data.xpos[bids]-data.xpos[keep[0]])@delta.T
  R=delta@data.xmat[bids].reshape(4,3,3)
  positions.append(p);rots.append(R)
  fh=[]
  for gids in footgeoms:
   minimum=np.inf
   for gid in gids:
    mid=model.geom_dataid[gid];a=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
    v=model.mesh_vert[a:a+n]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
    v=rootP+(v-data.xpos[keep[0]])@delta.T;minimum=min(minimum,v[:,2].min())
   fh.append(minimum)
  floors.append(fh)
 p=np.asarray(positions);R=np.asarray(rots);f=np.asarray(floors)
 error=np.linalg.norm(p[:,:2]-target,axis=-1)*100
 angle=np.rad2deg(np.arccos(np.clip((np.einsum('thij,thij->th',R[:,:2]@local,semantic)-1)/2,-1,1)))
 jointdiff=np.linalg.norm(p-(d['world_position_cm'][:,ids]@C)/100,axis=-1)
 report[version]=dict(wrist_position_cm=stats(error),wrist_orientation_deg=stats(angle),minimum_native_foot_mesh_cm=float(f.min()*100),foot_penetration_frames=int((f<-.001).any(1).sum()),native_descriptor_endpoint_difference_mm=stats(jointdiff*1000),joint_limit_violation_rad=max(allbounds),root_vertical_acceleration_m_s2=stats(np.abs(np.diff(d['root_t'][:,1],n=2)*9)),joint_acceleration_rad_s2=stats(np.abs(np.diff(d['q'],n=2,axis=0)*900)),wrist_tolerance_all_frames=bool((error<=1).all() and (angle<=15).all()),foot_floor_m=f.tolist(),wrist_errors_cm=error.tolist(),wrist_errors_deg=angle.tolist())
out=root/'observed-refinement'/label/('native-baseline.json' if '--before-only' in sys.argv else 'native-validation.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
print(json.dumps({version:{k:v for k,v in r.items() if k not in ['foot_floor_m','wrist_errors_cm','wrist_errors_deg']} for version,r in report.items()},indent=2))
