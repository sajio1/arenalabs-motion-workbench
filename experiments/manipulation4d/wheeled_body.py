from pathlib import Path
import json,time,hashlib
import numpy as np,mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from support_reference import support_reference
POLICY='native-shoulder-frame-contact-velocity-bounded-six-v11'
def solve(row,bodywork,out,targets,desired,offsets):
 bodywork=Path(bodywork);out=Path(out);m=mujoco.MjModel.from_xml_path(row['xml']);meta=json.loads(Path(row['metadata']).read_text());h=np.load(bodywork/'smplx_motion.npz');n=len(targets)
 source_hashes={k:hashlib.sha256((bodywork/k).read_bytes()).hexdigest() for k in ['canonical.npz','observations.npz','smplx_motion.npz','operation-accuracy/source-targets.npz']}
 from wheeled_prior import prepare
 raw=prepare(row,bodywork,out,targets,desired,offsets);rawhash=hashlib.sha256(raw.read_bytes()).hexdigest();prior=np.load(raw)
 floor,_,_,ground=support_reference(h,np.load(bodywork/'SMPLX_NEUTRAL.npz',allow_pickle=True));basez=floor+meta['ground_offset']
 basis=np.array([[0,0,1],[1,0,0],[0,1,0.]])
 rootR=h['global_rotation'][:,0]@basis.T;yaw=np.unwrap(np.arctan2(rootR[:,1,0],rootR[:,0,0]));baseR=Rotation.from_euler('z',yaw).as_matrix()
 root=np.c_[h['joints'][:,0,:2],np.full(n,basez)]
 ids=[m.body(w).id for w in meta['wrists']];shoulders=[m.body(s).id for s in row['mobile_semantics']['shoulders']];elbows=[m.body(s).id for s in row['mobile_semantics']['elbows']];torso=m.body(meta['torso']).id
 data=mujoco.MjData(m);mujoco.mj_forward(m,data);length=np.linalg.norm(data.xpos[elbows]-data.xpos[shoulders],axis=-1)
 human_offset=np.einsum('tji,tj->ti',baseR,h['joints'][:,[16,17]].mean(1)-h['joints'][:,0])
 robot_offset=data.xpos[shoulders].mean(0)-data.xpos[1]
 alignment=np.r_[np.median(human_offset[:,:2],axis=0)-robot_offset[:2],0.]
 root[:,:2]+=np.einsum('tij,j->ti',baseR,alignment)[:,:2]
 (out/'root-frame-alignment.json').write_text(json.dumps(dict(translation_local_m=alignment.tolist(),policy='Constant anatomical-to-native root-frame translation from shoulder center; no per-frame optimized base offset',human_offset_m=np.median(human_offset,axis=0).tolist(),native_neutral_shoulder_offset_m=robot_offset.tolist()),indent=2))
 resttorso=data.xmat[torso].reshape(3,3).copy();chest=h['global_rotation'][:,9]@basis.T@resttorso
 jointmeta={j['name']:j for j in meta['joints']};lo=m.jnt_range[1:,0];hi=m.jnt_range[1:,1];v=np.array([float(jointmeta[m.joint(j).name]['limits']['velocity']) for j in range(1,m.njnt)])
 nq=len(lo);columns=np.array([m.jnt_dofadr[j] for j in range(1,m.njnt)])
 upper=h['joints'][:,[18,19]]-h['joints'][:,[16,17]];upper/=np.maximum(np.linalg.norm(upper,axis=-1,keepdims=True),1e-10)
 seed=np.clip(prior['seed'].copy(),lo,hi)
 # Anatomical shoulder center matches shoulder center, not waist_Link_3.
 for frame in range(n):
  data.qpos[:3]=root[frame];data.qpos[3:7]=Rotation.from_matrix(baseR[frame]).as_quat()[[3,0,1,2]];data.qpos[7:]=seed[frame];mujoco.mj_forward(m,data)
  for j in range(1,m.njnt):
   if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:
    axis=data.xaxis[j];cor=np.dot(h['joints'][frame,[16,17]].mean(0)-data.xpos[shoulders].mean(0),axis)
    seed[frame,m.jnt_qposadr[j]-7]=np.clip(seed[frame,m.jnt_qposadr[j]-7]+cor,m.jnt_range[j,0],m.jnt_range[j,1])
 head=m.body(row['mobile_semantics']['head']).id
 neutral=mujoco.MjData(m);mujoco.mj_forward(m,neutral)
 headgoal=h['global_rotation'][:,15]@basis.T@neutral.xmat[head].reshape(3,3)
 def ancestors(b):
  chain=[]
  while b:chain.append(b);b=int(m.body_parentid[b])
  return chain
 def arm(b):
  chain=ancestors(b)
  return next((i for i,sh in enumerate(shoulders) if sh in chain),-1)
 arm_side=[arm(b) for b in range(m.nbody)]
 collision_pairs=[]
 for a in range(m.ngeom):
  if not (m.geom_contype[a] or m.geom_conaffinity[a]):continue
  ba=int(m.geom_bodyid[a])
  for b in range(a+1,m.ngeom):
   if not (m.geom_contype[b] or m.geom_conaffinity[b]):continue
   bb=int(m.geom_bodyid[b]);sa=arm_side[ba];sb=arm_side[bb]
   if sa==sb or max(sa,sb)<0:continue
   if ba in ancestors(bb)[:3] or bb in ancestors(ba)[:3]:continue
   collision_pairs.append((a,b))
 collision_index={pair:i for i,pair in enumerate(collision_pairs)}
 qout=[];history=[];started=time.time()
 def skew(x):return np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]],float)
 def jacbody(b,point=None):
  jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,data,jp,jr,data.xpos[b] if point is None else point,b);return jp[:,columns],jr[:,columns]
 for frame in range(n):
  data.qpos[:3]=root[frame];data.qpos[3:7]=Rotation.from_matrix(baseR[frame]).as_quat()[[3,0,1,2]]
  prev=qout[-1] if frame else None
  lower=lo if prev is None else np.maximum(lo,prev-v/30)
  higher=hi if prev is None else np.minimum(hi,prev+v/30)
  cache={}
  def calc(q):
   if 'q' in cache and np.array_equal(q,cache['q']):return cache['res'],cache['jac']
   data.qpos[7:]=q;mujoco.mj_forward(m,data);res=[];jac=[]
   for side,b in enumerate(ids):
    R=data.xmat[b].reshape(3,3);point=data.xpos[b]+R@offsets[frame,side];jp,jr=jacbody(b,point)
    res.extend((point-targets[frame,side])/.01);jac.extend(jp/.01)
    for col in range(3):
     res.extend((R[:,col]-desired[frame,side,:,col])/.1);jac.extend(-skew(R[:,col])@jr/.1)
    sh=shoulders[side];el=elbows[side];jsh,_=jacbody(sh);jel,_=jacbody(el)
    res.extend((data.xpos[el]-data.xpos[sh]-upper[frame,side]*length[side])/.08);jac.extend((jel-jsh)/.08)
   j0,_=jacbody(shoulders[0]);j1,_=jacbody(shoulders[1]);center=data.xpos[shoulders].mean(0);goal=h['joints'][frame,[16,17]].mean(0)
   res.append((center[2]-goal[2])/.1);jac.append((j0[2]+j1[2])/.2)
   R=data.xmat[torso].reshape(3,3);_,jr=jacbody(torso)
   for col in range(3):res.extend((R[:,col]-chest[frame,:,col])/.8);jac.extend(-skew(R[:,col])@jr/.8)
   R=data.xmat[head].reshape(3,3);_,jr=jacbody(head)
   for col in range(3):res.extend((R[:,col]-headgoal[frame,:,col])/.4);jac.extend(-skew(R[:,col])@jr/.4)
   cr=np.zeros(len(collision_pairs));cj=np.zeros((len(collision_pairs),nq))
   for contact in data.contact:
    pair=tuple(sorted(map(int,contact.geom)))
    if pair not in collision_index or contact.dist>=0:continue
    index=collision_index[pair];depth=-float(contact.dist)
    if depth<=cr[index]:continue
    cr[index]=depth;normal=contact.frame[:3];a=int(m.geom_bodyid[contact.geom[0]]);b=int(m.geom_bodyid[contact.geom[1]])
    ja,_=jacbody(a,contact.pos);jb,_=jacbody(b,contact.pos);cj[index]=-normal@(jb-ja)
   res.extend(cr/.01);jac.extend(cj/.01)
   # Model prior remains a weak secondary objective; no decoder position stream.
   scale=np.maximum(hi-lo,.1)*3
   res.extend((q-seed[frame])/scale);jac.extend(np.diag(1/scale))
   if prev is not None:
    scale=np.maximum(v/30,.001)*4
    res.extend((q-prev)/scale);jac.extend(np.diag(1/scale))
   cache.update(q=q.copy(),res=np.array(res),jac=np.array(jac));return cache['res'],cache['jac']
  starts=[np.clip(seed[frame],lower,higher)] if prev is not None else [seed[frame],np.clip(m.qpos0[7:],lo,hi)]
  if prev is not None:starts=[np.clip(prev,lower,higher)]
  if prev is None:
   from scipy.stats import qmc
   starts += [lo+(hi-lo)*u for u in qmc.Sobol(nq,scramble=False).random_base2(4)[1:]]
  results=[least_squares(lambda x:calc(x)[0],x,jac=lambda x:calc(x)[1],bounds=(lower,higher),max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7) for x in starts]
  if prev is None:print("INITIAL_CANDIDATES",[float(r.cost) for r in results],flush=True)
  result=min(results,key=lambda r:r.cost);qout.append(result.x);history.append(dict(frame=frame,success=bool(result.success),cost=float(result.cost),nfev=int(result.nfev)))
  if frame%50==0:print('FRAME',frame,'cost',result.cost,flush=True)
 def output(q,path):
  positions=[];rots=[];actual=[];angles=[];qpos=[]
  for frame in range(n):
   data.qpos[:3]=root[frame];data.qpos[3:7]=Rotation.from_matrix(baseR[frame]).as_quat()[[3,0,1,2]];data.qpos[7:]=q[frame];mujoco.mj_forward(m,data)
   R=data.xmat[ids].reshape(2,3,3);a=data.xpos[ids]+np.einsum('sij,sj->si',R,offsets[frame]);actual.append(a);angles.append(Rotation.from_matrix(R@desired[frame].transpose(0,2,1)).magnitude());positions.append(data.xpos.copy());rots.append(data.xmat.reshape(-1,3,3).copy());qpos.append(data.qpos.copy())
  actual=np.array(actual);err=np.linalg.norm(actual-targets,axis=-1)*1000
  np.savez_compressed(path,qpos=qpos,actual=actual,target=targets,orientation_error_rad=angles,ground_height_m=floor,positions=positions,rotations=rots)
  return dict(mean_mm=float(err.mean()),max_mm=float(err.max()),p95_mm=float(np.percentile(err,95)),orientation_mean_deg=float(np.rad2deg(angles).mean()))
 before=output(seed,out/'unrefined-motion.npz');qout=np.array(qout);metrics=output(qout,out/'wheeled-motion.npz')
 report=dict(policy=POLICY,seconds=time.time()-started,source_hashes=source_hashes,raw_sha256=rawhash,source_trajectory_edited=False,ground=ground,mean_tcp_mm=metrics['mean_mm'],max_tcp_mm=metrics['max_mm'],quality=metrics,unrefined=before,optimizer_success=all(x['success'] for x in history),message='Per-frame native least squares with hard joint velocity bounds',velocity_violation=float(np.max(abs(np.diff(qout,axis=0))*30-v)),base_policy='SMPL pelvis motion with constant derived shoulder-center frame alignment',wheel_actuation='Fixed vendor joints; no wheel control',physical_validation=False,base_travel_m=float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=-1).sum()),source_anatomy='Native shoulder center corresponds to human shoulder center',collision_constraints=True,collision_policy="Vendor native collision meshes, arm/trunk and opposite-arm penetration penalties; not physical validation")
 assert rawhash==hashlib.sha256(raw.read_bytes()).hexdigest();assert all(v==hashlib.sha256((bodywork/k).read_bytes()).hexdigest() for k,v in source_hashes.items())
 (out/'wheeled-validation.json').write_text(json.dumps(report,indent=2));(out/'solver-frames.json').write_text(json.dumps(history));print(report,flush=True);return out/'wheeled-motion.npz'
