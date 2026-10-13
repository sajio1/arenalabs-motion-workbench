"""Experimental observed-wrist/body-prior solve. Does not modify production.

All robots share this policy. Original wrists are absolute metric targets;
SMPL-X supplies lower-body posture and motion, never replacement observations.
"""
from pathlib import Path
import os,sys,json,hashlib,time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.physical_fk import PhysicalFK
from greenwich_pipelines.pipelines.nymeria.sole_geometry import sole_surface
from greenwich_umi_proof.palm_frames import palm_frame,robot_palm_local
from alphamotion.engine.nets.rotations import rot6d_to_matrix,matrix_to_rot6d

root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent))
label=sys.argv[1];start=time.time();device='cuda';torch.set_num_threads(4)
cast=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64)
C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
row=json.loads((root/(label+'-robot.json')).read_text())[0]
body=load_robot(row);fk=PhysicalFK(body,device,torch.float64)
path=next((root/(label+'-result')).rglob('motion.npz'))
d=dict(np.load(path));raw_path=path.with_name('prime_original.npz')
raw_hash=hashlib.sha256(raw_path.read_bytes()).hexdigest()
obs=np.load(root/'observations.npz');human=np.load(root/'smplx_motion.npz')
names=list(body.spec.joint_names);n=len(d['q'])
ids=[names.index(row['endpoints'][r]) for r in ('left_wrist','right_wrist','left_ankle','right_ankle')]
target=cast(obs['wrist_position']@C.T)
semantic=C@palm_frame(obs['palm_forward'],obs['palm_normal'])
local=np.stack([robot_palm_local(row['palm_frames'],s+'_wrist') for s in ('left','right')])
desired=cast(semantic@local.transpose(0,2,1)[None])
q0=cast(d['q']).flatten(1)[:,fk.active_indices].detach()
g0=cast(d['root_t']/100)
r0=cast(d['rot6d'][:,fk.root])
u=q0.clone().requires_grad_();g=g0.clone().requires_grad_();r=r0.clone().requires_grad_()
priorR=rot6d_to_matrix(cast(d['raw_rot6d'])).detach()
priorpos=fk.positions(priorR).detach()
sole,hull,valid=sole_surface(body,ids[2:]);hull=cast(hull);valid=torch.as_tensor(valid,device=device)
hj=human['joints']@C.T
# Knee flexion is a morphology-independent angle, resolved from the physical chain.
chains=[]
for foot in ids[2:]:
 chain=[];j=foot
 while j!=fk.root:
  chain.append(j);j=int(body.spec.parents[j])
 # Fold colocated joint axes into unique physical landmarks, distal first.
 unique=[]
 neutral=fk.positions(cast(np.broadcast_to(np.eye(3),(1,len(names),3,3)))).cpu().numpy()[0]
 for j in chain:
  if not unique or np.linalg.norm(neutral[j]-neutral[unique[-1]])>.08:unique.append(j)
 if len(unique)<3:raise ValueError(('No knee chain',chain,names))
 chains.append(unique[:3])
print('CHAINS',[[names[j] for j in ch] for ch in chains],flush=True)
def flex(p,a,b,c):
 v=p[:,a]-p[:,b];w=p[:,b]-p[:,c]
 return torch.atan2(torch.cross(v,w,dim=-1).norm(dim=-1),(v*w).sum(-1))
humanp=cast(hj)
knee_target=torch.stack([flex(humanp,7,4,1),flex(humanp,8,5,2)],1).detach()
# Resolve the signed hinge branch from native limits and FK. An unsigned bone
# angle alone admits an equally good hyperextended pose and branch discontinuities.
knee_axes=[];knee_q=[];calibration=[]
with torch.no_grad():
 for side,chain in enumerate(chains):
  candidates=np.flatnonzero((fk.active_indices.cpu().numpy()//3)==chain[1])
  if len(candidates)!=1:raise ValueError('Expected one physical knee hinge; ambiguous topology')
  axis=int(candidates[0]);low=float(fk.lower[axis]);high=float(fk.upper[axis])
  if not np.isfinite([low,high]).all():raise ValueError('Knee calibration requires finite mechanical limits')
  grid=np.linspace(low,high,801);probe=torch.zeros((len(grid),len(fk.active_indices)),device=device,dtype=torch.float64)
  probe[:,axis]=cast(grid);_,pp=fk(probe,cast(np.broadcast_to(np.eye(3),(len(grid),3,3)).copy()))
  bend=flex(pp,*chain).cpu().numpy();straight=int(np.argmin(bend))
  branch=np.arange(straight,len(grid)) if bend[-1]>=bend[0] else np.arange(straight,-1,-1)
  if np.any(np.diff(bend[branch]) < -1e-5):raise ValueError('Non-monotonic knee calibration')
  target_q=np.interp(knee_target[:,side].cpu().numpy(),bend[branch],grid[branch])
  knee_axes.append(axis);knee_q.append(target_q)
  calibration.append(dict(joint=names[chain[1]],axis=axis,straight_q_deg=float(np.rad2deg(grid[straight])),branch_end_q_deg=float(np.rad2deg(grid[branch[-1]])),minimum_flex_deg=float(np.rad2deg(bend[straight])),maximum_flex_deg=float(np.rad2deg(bend[branch[-1]]))))
 knee_q=cast(np.stack(knee_q,1));u[:,knee_axes]=knee_q
print('KNEE_CALIBRATION',json.dumps(calibration),flush=True)
foot_target=cast(hj[:,[7,8]])
# Confidence, not a binary flat-foot lock; source low/slow phases, temporally blended.
heights=hj[:,[7,8],1];speed=np.linalg.norm(np.gradient(hj[:,[7,8]],axis=0)*30,axis=-1)
base=np.percentile(heights,15,axis=0)
conf=np.exp(-np.maximum(heights-base-.025,0)**2/.045**2)*np.exp(-speed**2/.35**2)
conf=cast(gaussian_filter1d(conf,1.5,axis=0))
clearance=cast(np.maximum(gaussian_filter1d(heights-base,1.5,axis=0)-.025,0))
# Reference root vertical dynamics come from the body prior, not prior floor snaps.
root_prior=g0.cpu().numpy().copy()
root_prior[:,1]=np.median(root_prior[:,1]-hj[:,0,1])+gaussian_filter1d(hj[:,0,1],1,axis=0)
root_prior=cast(root_prior)
def d1(x):return x[1:]-x[:-1]
def d2(x):return x[2:]-2*x[1:-1]+x[:-2]
history=[];evaluations=0
def evaluate():
 R,p=fk(u,rot6d_to_matrix(r));world=p+g[:,None]
 e=world[:,ids[:2]]-target
 delta=R[:,ids[:2]]@desired.transpose(-1,-2)
 skew=torch.stack((delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]),-1)*.5
 angle=torch.atan2(skew.norm(dim=-1),.5*(delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1))
 hw=torch.einsum('tfij,fkj->tfki',R[:,ids[2:]],hull)+world[:,ids[2:],None]
 heights=torch.where(valid[None],hw[...,1],torch.inf).amin(-1)
 knees=torch.stack([flex(world,*chain) for chain in chains],1)
 terms={
  'observed_wrist_position':(torch.relu(e.norm(dim=-1)-.002)/.0015).square().mean(),
  'observed_wrist_orientation':(torch.relu(angle-.035)/.075).square().mean(),
  'model_rotation':((R-priorR)/.7).square().mean(),
  'model_relative_position':((p-priorpos)/.12).square().mean(),
  'feet_horizontal':((world[:,ids[2:]][...,[0,2]]-foot_target[...,[0,2]])/.065).square().mean(),
  'knee_prior':((u[:,knee_axes]-knee_q)/.12).square().mean(),
  'clearance':((heights-clearance)/.04).square().mean(),
  'support':(conf*(heights/.015).square()).mean(),
  'penetration':(torch.relu(.001-hw[...,1])[:,valid]/.0003).square().mean(),
  'root_prior':((g-root_prior)/.12).square().mean(),
  'joint_correction_velocity':(d1(u-q0)/.12).square().mean(),
  'joint_acceleration':(d2(u)/.025).square().mean(),
  'root_correction_acceleration':(d2(g-root_prior)/.0015).square().mean(),
  'root_rotation_acceleration':(d2(R[:,fk.root])/.025).square().mean(),
 }
 return sum(terms.values()),terms,R,world,heights,knees,e,angle
parameters=[u,g,r];sizes=[p.numel() for p in parameters]
def closure(x):
 global evaluations
 offset=0
 with torch.no_grad():
  for p,s in zip(parameters,sizes):p.copy_(cast(x[offset:offset+s]).reshape(p.shape));p.grad=None;offset+=s
 loss,terms,*_=evaluate()
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
 loss.backward();evaluations+=1
 if evaluations==1 or evaluations%100==0:
  entry=dict(evaluations=evaluations,loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()})
  history.append(entry);print(json.dumps(entry),flush=True)
 return float(loss.detach()),np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in parameters])
x0=np.concatenate([p.detach().cpu().numpy().ravel() for p in parameters])
lo=np.r_[np.tile(fk.lower.cpu().numpy(),n),np.full(sizes[1]+sizes[2],-np.inf)]
hi=np.r_[np.tile(fk.upper.cpu().numpy(),n),np.full(sizes[1]+sizes[2],np.inf)]
res=minimize(closure,x0,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(maxiter=1600,maxcor=30,ftol=1e-11,gtol=1e-7,maxls=30))
closure(res.x)
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
with torch.no_grad():
 loss,terms,R,w,h,k,e,a=evaluate();array=lambda x:x.detach().cpu().numpy()
 out=dict(d);out.update(q=array(fk.expand_angles(u)),rot6d=array(matrix_to_rot6d(R)),root_t=array(g)*100,world_position_cm=array(w)*100,sole_surface_height_cm=array(h)*100,support_surface_height_cm=array(h)*100,observed_wrist_position_yup_m=array(target),observed_wrist_rotation_yup=array(desired),wrist_error_m=array(e.norm(dim=-1)),wrist_error_deg=array(a)*180/np.pi,knee_flexion_rad=array(k),source_knee_flexion_rad=array(knee_target),refiner='observed-wrists-body-prior-v3')
 dest=root/'observed-refinement'/label;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'motion.npz',**out)
 report=dict(knee_calibration=calibration,robot=label,seconds=time.time()-start,source='original Nymeria wrist observations; SMPL-X lower-body prior',raw_sha256=raw_hash,original_prior_unchanged=hashlib.sha256(raw_path.read_bytes()).hexdigest()==raw_hash,iterations=int(res.nit),optimizer_success=bool(res.success),message=str(res.message),wrist_position_error_cm=stats(array(e.norm(dim=-1))*100),wrist_orientation_error_deg=stats(array(a)*180/np.pi),minimum_foot_height_cm=float(h.min())*100,knee_range_deg=np.rad2deg(np.stack([array(k).min(0),array(k).max(0)])).tolist(),prior_knee_range_deg=np.rad2deg(np.stack([array(knee_target).min(0),array(knee_target).max(0)])).tolist(),root_vertical_acceleration_m_s2=stats(np.abs(np.diff(array(g)[:,1],n=2)*900)),terms={key:float(v) for key,v in terms.items()},history=history,physical_validation=False)
 (dest/'report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps({key:v for key,v in report.items() if key!='history'}),flush=True)
