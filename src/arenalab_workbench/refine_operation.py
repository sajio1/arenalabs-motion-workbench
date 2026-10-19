"""Production observed-reference/body-prior refinement shared by robot adapters.

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
from arenalab_workbench.body_frames import BodyFrames,angle as frame_angle

root=Path(os.environ.get('ARENA_EGO_WORKDIR',Path(__file__).resolve().parent))
label=sys.argv[1];start=time.time();device='cuda';torch.set_num_threads(4)
policy=dict(body_scale=.08,wrist_orientation_weight=1.,knee_weight=1.,root_height_weight=1.,model_weight=1.)
# One shared policy for both robots and both reference-point tests.
knee_bias=torch.zeros((),device=device,dtype=torch.float64,requires_grad=True)
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
body_frames=BodyFrames(body,fk,ids[:2],ids[2:])
human_chest=cast(C@human['global_rotation'][:,9])
human_pelvis=cast(C@human['global_rotation'][:,0])
human_relative=human_pelvis.transpose(-1,-2)@human_chest
print('BODY_FRAMES',json.dumps(body_frames.metadata),flush=True)
target=cast(obs['wrist_position']@C.T)
mode=os.environ.get('ARENA_OPERATION_MODE','palm');point_index={'palm':0,'pinch':6,'tcp':0}[mode]
operation_targets=cast(np.load(root/'operation-accuracy/source-targets.npz')['points'][:,:,point_index]@C.T)
operation_local=cast(np.load(root/'operation-accuracy'/f'{label}-local-points.npz')['points'][:,:,point_index]@C.T)

semantic=C@palm_frame(obs['palm_forward'],obs['palm_normal'])
local=np.stack([robot_palm_local(row['palm_frames'],s+'_wrist') for s in ('left','right')])
desired=cast(semantic@local.transpose(0,2,1)[None])
# Optional calibrated tool-frame conversion; same orientation objective and weights.
point_data=np.load(root/'operation-accuracy'/f'{label}-local-points.npz')
if 'target_rotation_zup' in point_data:
 desired=cast(C@point_data['target_rotation_zup']@C.T)
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
knee_sign=cast([np.sign(c['branch_end_q_deg']-c['straight_q_deg']) for c in calibration])
knee_branch_low=cast([min(c['straight_q_deg'],c['branch_end_q_deg'])*np.pi/180 for c in calibration])
knee_branch_high=cast([max(c['straight_q_deg'],c['branch_end_q_deg'])*np.pi/180 for c in calibration])
# Geometry-derived support plane; raw source and model priors remain immutable.
from arenalab_workbench.support_reference import support_reference
floor_height, support_confidence, support_clearance, support_metadata = support_reference(human, np.load(Path(os.environ.get('ARENA_SOURCE_WORK',str(root)))/'SMPLX_NEUTRAL.npz',allow_pickle=True))
source_manifest=json.loads((Path(os.environ.get('ARENA_SOURCE_WORK',str(root)))/'input_manifest.json').read_text())
height_calibration=source_manifest.get('height_calibration')
if height_calibration:
 floor_height=float(height_calibration['floor_z_m'])
 support_metadata.update(ground_height_m=floor_height,ground_status='User-declared trajectory minimum height; estimated floor',height_calibration=height_calibration)
conf=cast(support_confidence);clearance=cast(support_clearance)
sole=cast(sole)
valid_indices=torch.where(valid)
print('SUPPORT_REFERENCE',json.dumps(support_metadata),flush=True)
# Reference root vertical dynamics come from the body prior, not prior floor snaps.
root_prior=g0.cpu().numpy().copy()
root_prior[:,1]=np.median(root_prior[:,1]-hj[:,0,1])+gaussian_filter1d(hj[:,0,1],1,axis=0)
root_prior=cast(root_prior)
def d1(x):return x[1:]-x[:-1]
def d2(x):return x[2:]-2*x[1:-1]+x[:-2]
root_weights=cast([1.,policy.get('root_height_weight',1.),1.])
history=[];evaluations=0
wrist_index=torch.tensor(ids[:2],device=device);ankle_index=torch.tensor(ids[2:],device=device);knee_index=torch.tensor(knee_axes,device=device);horizontal_index=torch.tensor([0,2],device=device)
def evaluate():
 R,p=fk(u,rot6d_to_matrix(r));world=p+g[:,None]
 e=world[:,wrist_index]-target
 operation_world=world[:,wrist_index]+torch.einsum('tsij,tsj->tsi',R[:,wrist_index],operation_local)
 operation_error=operation_world-operation_targets
 delta=R[:,wrist_index]@desired.transpose(-1,-2)
 skew=torch.stack((delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]),-1)*.5
 angle=torch.atan2(skew.norm(dim=-1),.5*(delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1))
 hw=torch.einsum('tfij,fkj->tfki',R[:,ankle_index],hull)+world[:,ankle_index,None]
 heights=torch.where(valid[None],hw[...,1]-floor_height,torch.inf).amin(-1)
 sole_world=torch.einsum('tfij,fkj->tfki',R[:,ankle_index],sole)+world[:,ankle_index,None]
 sole_height=sole_world[...,1]-floor_height
 knees=torch.stack([flex(world,*chain) for chain in chains],1)
 chest,relative,pelvis=body_frames(R)
 terms={
  'human_chest_orientation':(torch.relu(frame_angle(chest,human_chest)-.0873)/policy['body_scale']).square().mean(),
  'human_chest_relative_pelvis':(torch.relu(frame_angle(relative,human_relative)-.0873)/.25).square().mean(),
  'observed_operation_position':(torch.relu(operation_error.norm(dim=-1)-.002)/.0015).square().mean(),
  'observed_wrist_orientation':policy.get('wrist_orientation_weight',1.)*(torch.relu(angle-.035)/.075).square().mean(),
  'model_rotation':policy.get('model_weight',1.)*((R-priorR)/.7).square().mean(),
  'model_relative_position':policy.get('model_weight',1.)*((p-priorpos)/.12).square().mean(),
  'feet_horizontal':((world[:,ankle_index][...,horizontal_index]-foot_target[...,horizontal_index])/.065).square().mean(),
  'knee_prior':policy.get('knee_weight',1.)*((u[:,knee_index]-torch.clamp(knee_q+knee_sign*knee_bias,min=knee_branch_low,max=knee_branch_high))/.12).square().mean(),
  'knee_morphology_bias':(knee_bias/.3).square(),
  'foot_correction_velocity':(torch.relu(d1(world[:,ankle_index]-foot_target)[...,horizontal_index].norm(dim=-1)-.0033)/.0033).square().mean(),
  'clearance':((heights-clearance)/.04).square().mean(),
  'support':(conf[...,None]*(sole_height/.015).square()).mean(),
  'support_plane':(conf*((sole_height.amax(-1)-sole_height.amin(-1))/.02).square()).mean(),
  'penetration':(torch.relu(floor_height+.001-hw[...,1])[:,valid_indices[0],valid_indices[1]]/.0003).square().mean(),
  'root_prior':(((g-root_prior)/.12).square()*root_weights).mean(),
  'joint_correction_velocity':(d1(u-q0)/.12).square().mean(),
  'joint_acceleration':(d2(u)/.018).square().mean(),
  'root_correction_acceleration':(d2(g-root_prior)/.0012).square().mean(),
  'root_rotation_acceleration':(d2(R[:,fk.root])/.025).square().mean(),
 }
 return sum(terms.values()),terms,R,world,heights,knees,e,angle
parameters=[u,g,r,knee_bias];sizes=[p.numel() for p in parameters]
backend=os.environ.get('ARENA_REFINEMENT_BACKEND','cuda_graph')
if backend not in {'cuda_graph','eager'}:raise ValueError('Unknown refinement execution backend')
captured=None
if backend=='cuda_graph':
 from arenalab_workbench.captured_objective import CapturedObjective
 captured=CapturedObjective(evaluate,parameters)

def closure(x):
 global evaluations
 if captured is not None:
  loss,gradient=captured(x);evaluations+=1
  if evaluations==1 or evaluations%100==0:
   entry=dict(evaluations=evaluations,loss=loss,terms=captured.term_values());history.append(entry);print(json.dumps(entry),flush=True)
  return loss,gradient
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
lo=np.r_[np.tile(fk.lower.cpu().numpy(),n),np.full(sizes[1]+sizes[2],-np.inf),-.5]
hi=np.r_[np.tile(fk.upper.cpu().numpy(),n),np.full(sizes[1]+sizes[2],np.inf),.5]
res=minimize(closure,x0,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(maxiter=1600,maxcor=30,ftol=1e-11,gtol=1e-7,maxls=30))
closure(res.x)
def stats(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max()))
with torch.no_grad():
 loss,terms,R,w,h,k,e,a=evaluate();array=lambda x:x.detach().cpu().numpy()
 out=dict(d);out.update(ground_height_m=np.array(floor_height),support_confidence=support_confidence,ground_estimated=np.array(True),q=array(fk.expand_angles(u)),rot6d=array(matrix_to_rot6d(R)),root_t=array(g)*100,world_position_cm=array(w)*100,sole_surface_height_cm=array(h)*100,support_surface_height_cm=array(h)*100,observed_wrist_position_yup_m=array(target),observed_wrist_rotation_yup=array(desired),wrist_error_m=array(e.norm(dim=-1)),wrist_error_deg=array(a)*180/np.pi,knee_flexion_rad=array(k),source_knee_flexion_rad=array(knee_target),refiner='observed-operation-reference-point-experiment',operation_reference=mode,operation_target_yup_m=array(operation_targets),operation_local_yup_m=array(operation_local))
 dest=Path(os.environ.get('ARENA_OUTPUTDIR',root/'observed-refinement'/label));dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'motion.npz',**out)
 report=dict(compute_backend=backend,support_reference=support_metadata,knee_morphology_bias_deg=float(knee_bias.detach())*180/np.pi,policy=policy,body_frames=body_frames.metadata,knee_calibration=calibration,robot=label,seconds=time.time()-start,source='Original observed operation landmarks; unchanged SMPL-X body prior',operation_reference=mode,raw_sha256=raw_hash,original_prior_unchanged=hashlib.sha256(raw_path.read_bytes()).hexdigest()==raw_hash,iterations=int(res.nit),optimizer_success=bool(res.success),message=str(res.message),wrist_position_error_cm=stats(array(e.norm(dim=-1))*100),wrist_orientation_error_deg=stats(array(a)*180/np.pi),minimum_foot_height_cm=float(h.min())*100,knee_range_deg=np.rad2deg(np.stack([array(k).min(0),array(k).max(0)])).tolist(),prior_knee_range_deg=np.rad2deg(np.stack([array(knee_target).min(0),array(knee_target).max(0)])).tolist(),root_vertical_acceleration_m_s2=stats(np.abs(np.diff(array(g)[:,1],n=2)*900)),terms={key:float(v) for key,v in terms.items()},history=history,physical_validation=False)
 (dest/'report.json').write_text(json.dumps(report,indent=2));print('DONE',json.dumps({key:v for key,v in report.items() if key!='history'}),flush=True)
