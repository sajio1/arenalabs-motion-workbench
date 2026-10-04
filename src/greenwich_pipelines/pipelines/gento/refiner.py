
"""Arm-priority baseline with 45-degree tilt deadband and reach-gated height."""
import argparse,json,hashlib,time
import numpy as np,torch,mujoco,h5py
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from . import source as g
from .wrist_alignment import proximal_origins, angles as wrist_bend_angles
ROOT=g.ROOT
P=dict(position_scale_m=.005,orientation_scale=.15,prior_rotation_weight=.2,
 velocity_weight=.005,acceleration_weight=.2,fps=30,base_fixed=True,
 torso_horizontal_scale_m=.1,torso_tilt_scale=.25,posture_weight=1.,
 com_margin_fraction=.1,com_edge_scale_fraction=.1,torso_vertical_velocity_weight=.02,
 torso_tilt_free_deg=45.,height_hold_scale_m=.001,adam_steps=120,lbfgs_steps=1000,
 candidate_steps=150,stagnation_window=50,stagnation_relative_improvement=1e-4,stagnation_patience=3,
 wrist_alignment_scale_rad=float(np.deg2rad(30.)),
 torso_direction_policy='forward pitch alone has 45 degree free band; lateral roll and yaw retain neutral task-facing reference',
 reach_fraction=.95,limit_margin_deg=2.,reach_error_m=.01,
 initialization='bounded temporal initializer with shared first-frame branch search; immutable Prime prior',
 base_placement='baseline XY, fixed yaw facing median wrist working area',head_policy='camera rotation only, neck-only fit within official limits',
 height_policy='initial height from sequence median ego camera height minus neutral head-to-torso geometry; release only after failed fixed-height tracking near arm reach or physical joint limit')
def solve(body,clip,version='arm_priority_height_multistart',raw_path=None,frames=None,head_position_reference=False,head_placement=False,global_height=False,solve_setback=False,head_motion_reference=False,numerical_resume=None,fixed_setback_m=None,initialization_path=None,shoulder_placement=False,scene_backoff_m=0.):
 started=time.perf_counter()
 folder=ROOT/version/body/clip;folder.mkdir(parents=True,exist_ok=True)
 baseline=ROOT/'arm_priority15s'/body/clip
 if initialization_path is None:raise ValueError('Explicit raw-prior numerical initialization required')
 old=dict(np.load(initialization_path))
 resume_report=None
 if numerical_resume is not None:
  old=dict(np.load(numerical_resume/'result.npz'))
  resume_report=json.loads((numerical_resume/'report.json').read_text())
 if raw_path is None:raise ValueError('Explicit immutable raw_path required')
 prior_path=raw_path
 raw=dict(np.load(prior_path))
 if frames is not None:
  old={k:(v[:frames] if v.ndim and len(v)==len(old['qpos']) else v) for k,v in old.items()}
  for k in ['prior_R','position','orientation']:raw[k]=raw[k][:frames]
 meta=json.loads((ROOT/'models'/body/'metadata.json').read_text());m=mujoco.MjModel.from_xml_path(meta['model'])
 geo=g.geometry(m,meta['torso']);torso=geo['torso'];ids=raw['wrist_ids'];n=len(old['qpos'])
 proximal=proximal_origins(m,ids)
 anchor=np.r_[old['qpos'][0,:2],2*np.arctan2(old['qpos'][0,6],old['qpos'][0,3])]
 work_center=np.median(raw['position'].mean(1),axis=0)
 delta=work_center[:2]-anchor[:2];anchor[2]=np.arctan2(delta[1],delta[0])
 fk=g.FK(m,'cpu',anchor,dtype=torch.float64);t=fk.t
 x=torch.nn.Parameter(t(old['qpos'][:,7:]));lo=t(m.jnt_range[1:,0]);hi=t(m.jnt_range[1:,1])
 jointmeta={j['name']:j for j in meta['joints']}
 vel=t([float(jointmeta[m.joint(j).name]['limits'].get('velocity',1)) for j in range(1,m.njnt)])
 pi=raw['prior_body_ids'];prior=t(raw['prior_R']);target=t(raw['position']);desired=t(raw['orientation']);sem=t(raw['semantic_local'])
 ee_local_np=raw.get('ee_local_position',np.zeros((len(ids),3)));ee_local=t(ee_local_np)
 native_target=raw['position']-np.einsum('nkij,kj->nki',raw['orientation']@raw['semantic_local'].transpose(0,2,1),ee_local_np)
 torsoxy=t(geo['torso_xy_rest']);up_local=t(geo['up_local']);mass=t(geo['mass']);ipos=t(geo['body_ipos']);eq=t(geo['hull'])
 headj=[j for j in range(m.njnt) if 'head' in m.joint(j).name.lower()]
 head=int(m.jnt_bodyid[headj[-1]])
 with h5py.File(str(raw['source'])) as h:
  cam=np.asarray(h['transforms/camera'][:n])
  source_shoulders=np.stack([h['transforms/'+s+'Arm'][:n,:3,3]@g.C.T for s in ['left','right']],axis=1)
 head_target=t(cam[:,:3,3]@g.C.T)
 head_offset=t(raw.get('head_reference_local_position',np.zeros(3)))
 anatomical_head_reference='head_reference_local_position' in raw
 camera_to_forward=np.array([[0,-1,0],[0,0,-1],[1,0,0.]])
 headgoal=g.C@cam[:,:3,:3]@camera_to_forward
 allowed=np.zeros(n,bool)
 armj=[];shoulders=[];reach=[]
 optimizer_diagnostics=[]
 rest=mujoco.MjData(m);mujoco.mj_forward(m,rest)
 headlocal=rest.xmat[head].reshape(3,3).T.copy()
 head_sem=t(headlocal);head_rotation_target=t(headgoal)
 # Set scene height before holding it. A prior refined height is not a
 # measurement of the operator's workspace. This uses source height and
 # official rigid geometry, identically for all embodiments and clips.
 camera_height=float(np.median((cam[:,:3,3]@g.C.T)[:,2]))
 rest_head_point=rest.xpos[head]+rest.xmat[head].reshape(3,3)@head_offset.cpu().numpy()
 head_torso_height=float(rest_head_point[2]-rest.xpos[torso,2])
 h0=camera_height-head_torso_height
 height_initial=h0
 height_variable=torch.nn.Parameter(t(h0))
 height_learning=global_height
 if head_placement:
  # One constant scene placement from source head location and neutral rigid
  # geometry. Preserve facing yaw; the chassis never moves during the clip.
  if anatomical_head_reference:
   facing=work_center[:2]-np.median(cam[:,:3,3]@g.C.T,axis=0)[:2]
   anchor[2]=np.arctan2(facing[1],facing[0])
  yaw_R=Rotation.from_euler('z',anchor[2]).as_matrix()
  rest_head_offset=yaw_R@(rest_head_point-rest.xpos[1])
  anchor[:2]=np.median(cam[:,:3,3]@g.C.T,axis=0)[:2]-rest_head_offset[:2]
  fk.base_anchor=t(anchor)
 if shoulder_placement:
  if solve_setback or fixed_setback_m is not None:raise ValueError('Measured shoulder placement is fixed, not optimized')
  shoulder_ids=[]
  for wrist in ids:
   b=int(wrist);chain=[]
   while b!=torso and b>1:
    if m.body_jntnum[b]:chain.append(b)
    b=int(m.body_parentid[b])
   shoulder_ids.append(chain[-1])
  source_center=np.median(source_shoulders.mean(1),axis=0)
  robot_offset=Rotation.from_euler('z',anchor[2]).as_matrix()@(rest.xpos[shoulder_ids].mean(0)-rest.xpos[1])
  anchor[:2]=source_center[:2]-robot_offset[:2]
  fk.base_anchor=t(anchor)
 if scene_backoff_m:
  if solve_setback or fixed_setback_m is not None:raise ValueError('Scene backoff is fixed, never optimized')
  anchor[:2]-=scene_backoff_m*np.array([np.cos(anchor[2]),np.sin(anchor[2])])
  fk.base_anchor=t(anchor)
 anchor_reference=anchor.copy()
 setback=torch.nn.Parameter(t(0.))
 placement_learning=solve_setback and fixed_setback_m is None
 if fixed_setback_m is not None:
  if not solve_setback:raise ValueError('Fixed setback uses the existing setback placement convention')
  anchor[:2]=anchor_reference[:2]-fixed_setback_m*np.array([np.cos(anchor[2]),np.sin(anchor[2])])
  fk.base_anchor=t(anchor)
 torso_basis=t(rest.xmat[torso].reshape(3,3).T.copy())
 for w in ids:
  b=int(w);chain=[]
  while b!=torso and b>1:
   if m.body_jntnum[b]:chain.append(b)
   b=int(m.body_parentid[b])
  chain=chain[::-1];shoulders.append(chain[0])
  armj.append([int(m.jnt_qposadr[j]-7) for b in chain for j in range(m.body_jntadr[b],m.body_jntadr[b]+m.body_jntnum[b])])
  reach.append(float(np.linalg.norm(np.diff(rest.xpos[chain],axis=0),axis=-1).sum()))
 def loss():
  if placement_learning:
   aa=t(anchor_reference)
   fk.base_anchor=torch.stack([aa[0]-setback*aa[2].cos(),aa[1]-setback*aa[2].sin(),aa[2]])
  p,r=fk(x)
  ee=p[:,ids]+(r[:,ids]@ee_local[None,...,None]).squeeze(-1)
  ep=((ee-target)/P['position_scale_m']).square().mean()
  er=((r[:,ids]@sem-desired)/P['orientation_scale']).square().mean()
  forearm=p[:,ids]-p[:,proximal]
  forearm=forearm/torch.linalg.vector_norm(forearm,dim=-1,keepdim=True).clamp_min(1e-12)
  # Only bending is penalized. Rotation around the forearm is unrestricted.
  wrist_alignment=((forearm-(r[:,ids]@sem)[...,0])/P['wrist_alignment_scale_rad']).square().sum(-1).mean()
  pr=(r[:,pi]-prior).square().mean()*P['prior_rotation_weight']
  dv=(x[1:]-x[:-1])*30/vel;da=dv[1:]-dv[:-1]
  temporal=P['velocity_weight']*dv.square().mean()+P['acceleration_weight']*da.square().mean()+torch.relu(dv.abs()-1).square().mean()
  if head_motion_reference:
   # Joint-space averaging can hide a fast Cartesian jump of an intermediate
   # linkage while the tracked head and wrists remain nearly stationary.
   # Apply the same physical-space continuity cost to every moving body.
   moving=np.flatnonzero(m.body_dofnum>0)
   moving=moving[moving!=1]
   second_difference=p[2:,moving]-2*p[1:-1,moving]+p[:-2,moving]
   temporal+=(second_difference/P['position_scale_m']).square().mean()
  relative=(r[:,1].transpose(-1,-2)@(p[:,torso]-p[:,1])[...,None]).squeeze(-1)
  up=(r[:,torso]@up_local[:,None]).squeeze(-1)
  # A free FORWARD bend is not a free sideways bend or torso twist.
  anatomical=r[:,torso]@torso_basis
  relative_R=r[:,1].transpose(-1,-2)@anatomical
  au=relative_R[:,:,2];af=relative_R[:,:,0]
  forward=torch.atan2(au[:,0],au[:,2])
  lateral=torch.atan2(au[:,1],au[:,2])
  yaw=torch.atan2(af[:,1],af[:,0])
  pitch_excess=torch.relu(forward-np.deg2rad(P['torso_tilt_free_deg']))+torch.relu(-forward)
  # Forward displacement accompanies an allowed forward lean. Penalizing it
  # independently would silently discourage the user's permitted motion.
  posture=(((relative[:,1:2]-torsoxy[1:2]) if head_position_reference else (relative[:,:2]-torsoxy))/P['torso_horizontal_scale_m']).square().mean()
  posture+=((pitch_excess.square()+lateral.square()+yaw.square())/P['torso_tilt_scale']**2).mean()
  cm=((p+(r@ipos[None,...,None]).squeeze(-1))*mass[None,:,None]).sum(1)
  local=(r[:,1].transpose(-1,-2)@(cm-p[:,1])[...,None]).squeeze(-1)
  edge=torch.relu(local[:,:2]@eq[:,:2].T+eq[:,2]+geo['radius']*P['com_margin_fraction'])/(geo['radius']*P['com_edge_scale_fraction'])
  h=p[:,torso,2];hold=((h-(height_variable if height_learning else h0))/P['height_hold_scale_m']).square()*t(~allowed)
  vert=((h[1:]-h[:-1])*30).square().mean()*P['torso_vertical_velocity_weight']
  # Camera origin is a soft head-position proxy, not a calibrated robot sensor.
  head_point=p[:,head]+(r[:,head]@head_offset[:,None]).squeeze(-1)
  head_scale=P['position_scale_m'] if anatomical_head_reference else .15
  head_cost=((head_point-head_target)/head_scale).square().mean() if head_position_reference else 0.
  if head_motion_reference:
   # Embodiment differences may introduce ONE constant head-position offset,
   # but cannot justify a time-varying deviation from observed head motion.
   if not anatomical_head_reference:
    displacement=head_point-head_target
    head_cost+=((displacement-displacement.mean(0,keepdim=True))/P['position_scale_m']).square().mean()
   head_cost+=((r[:,head]@head_sem-head_rotation_target)/P['orientation_scale']).square().mean()
  return ep+er+pr+temporal+posture*P['posture_weight']+edge.square().mean()+hold.mean()+vert+head_cost+wrist_alignment
 def optimize(stage,coarse=False,warm=False):
  stage_start=time.perf_counter()
  parameters=[x,height_variable] if height_learning else [x]
  if placement_learning:parameters.append(setback)
  opt=torch.optim.Adam(parameters,lr=.004)
  for i in range(0 if warm or numerical_resume is not None else P['adam_steps']):
   opt.zero_grad();l=loss();l.backward();opt.step()
   with torch.no_grad():
    x.clamp_(lo,hi)
    if placement_learning:setback.clamp_(min=0.)
   if i%200==0:print(body,stage,i,float(l),flush=True)
  # Enforce official bounds inside the optimizer, not by clipping an
  # unconstrained solution after temporal optimization.
  sizes=[v.numel() for v in parameters]
  initial=np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in parameters]).astype(float)
  bounds=list(zip(np.tile(m.jnt_range[1:,0],n),np.tile(m.jnt_range[1:,1],n)))
  if height_learning:bounds.append((None,None))
  if placement_learning:bounds.append((0.,None))
  def assign(v):
   offset=0
   with torch.no_grad():
    for pvar,size in zip(parameters,sizes):
     pvar.copy_(t(v[offset:offset+size]).reshape(pvar.shape));offset+=size
  evaluations=0;cache={};history=[];stagnant=0;iterations=0
  def objective(v):
   nonlocal evaluations
   evaluations+=1
   assign(v)
   for pvar in parameters:pvar.grad=None
   value=loss();value.backward()
   gradient=np.concatenate([pvar.grad.detach().cpu().numpy().reshape(-1) for pvar in parameters]).astype(float)
   cache['x']=v.copy();cache['f']=float(value.detach())
   return cache['f'],gradient
  def callback(intermediate_result):
   nonlocal stagnant,iterations
   iterations+=1
   if iterations%P['stagnation_window']:return
   history.append(float(intermediate_result.fun))
   if len(history)>1:
    improvement=(history[-2]-history[-1])/max(abs(history[-2]),1.)
    stagnant=stagnant+1 if improvement<P['stagnation_relative_improvement'] else 0
    if stagnant>=P['stagnation_patience']:raise StopIteration
  result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',callback=callback,
   options={'maxiter':P['candidate_steps'] if coarse else P['lbfgs_steps'],'maxls':20,'ftol':1e-10,'gtol':1e-6})
  assign(result.x)
  optimizer_diagnostics.append(dict(stage=stage,iterations=int(result.nit),evaluations=evaluations,seconds=time.perf_counter()-stage_start,success=bool(result.success),message=str(result.message),objective=float(result.fun)))
  print(body,stage,'bounded optimizer',result.message,'objective',result.fun,flush=True)
 # Compare a retained numerical branch and a neutral branch by the SAME
 # objective. No action or embodiment determines which branch is selected.
 from .temporal_seed import initialize
 seed_diagnostics=None
 if numerical_resume is None:
  numeric,seed_diagnostics=initialize(m,raw,anchor,geo,P,head,head_offset.numpy(),headgoal,head_target.numpy(),n)
  with torch.no_grad():
   x.copy_(t(numeric));seed_p,_=fk(x);height_initial=float(seed_p[:,torso,2].median())
 original_start=x.detach().clone()
 candidates=[];candidate_costs={};candidate_heights=[];candidate_setbacks=[]
 for seed_name in (['numerical_continuation'] if numerical_resume is not None else ['bounded_temporal_initializer']):
  with torch.no_grad():
   x.copy_(original_start)
   height_variable.fill_(height_initial)
   setback.zero_()
   if resume_report is not None:
    height_variable.fill_(resume_report['height_reference_m'])
    setback.fill_(resume_report['solved_setback_m'])
   if fixed_setback_m is not None:setback.fill_(fixed_setback_m)
  optimize('height-held-'+seed_name,warm=True)
  with torch.no_grad():cost=float(loss())
  candidate_costs[seed_name]=cost
  candidates.append(x.detach().clone())
  candidate_heights.append(float(height_variable.detach()))
  candidate_setbacks.append(float(setback.detach()))
 chosen=int(np.argmin(list(candidate_costs.values())))
 with torch.no_grad():
  x.copy_(candidates[chosen]);height_variable.fill_(candidate_heights[chosen]);setback.fill_(candidate_setbacks[chosen])
 if global_height:h0=float(height_variable.detach())
 height_learning=False
 placement_learning=False
 chosen_setback=float(setback.detach())
 if solve_setback:
  anchor[:2]=anchor_reference[:2]-chosen_setback*np.array([np.cos(anchor_reference[2]),np.sin(anchor_reference[2])])
  fk.base_anchor=t(anchor)
 print(body,'seed objectives',candidate_costs,'selected',list(candidate_costs)[chosen],flush=True)
 with torch.no_grad():p,r=fk(x)
 pp=p.cpu().numpy();xx=x.detach().cpu().numpy()
 trial_ee=pp[:,ids]+np.einsum('nkij,kj->nki',r.cpu().numpy()[:,ids],ee_local_np)
 e=np.linalg.norm(trial_ee-raw['position'],axis=-1)
 ratio=np.linalg.norm(native_target-pp[:,shoulders],axis=-1)/np.array(reach)
 limit=np.zeros((n,2),bool)
 for k,js in enumerate(armj):
  limit[:,k]=(np.minimum(xx[:,js]-m.jnt_range[1:,0][js],m.jnt_range[1:,1][js]-xx[:,js])<=np.deg2rad(P['limit_margin_deg'])).any(1)
 allowed=((e>P['reach_error_m'])&((ratio>P['reach_fraction'])|limit)).any(1)
 if allowed.any():optimize('reach-gated',warm=True)
 with torch.no_grad():p,r=fk(x)
 pp=p.cpu().numpy();rr=r.cpu().numpy();q=np.broadcast_to(old['qpos'][0],old['qpos'].shape).copy();q[:,:2]=anchor[:2];q[:,7:]=x.detach().cpu().numpy();q[:,3:7]=[np.cos(anchor[2]/2),0,0,np.sin(anchor[2]/2)]
 data=mujoco.MjData(m)
 headj=[j for j in range(m.njnt) if 'head' in m.joint(j).name.lower()]
 head=int(m.jnt_bodyid[headj[-1]]);headadr=m.jnt_qposadr[headj]
 mujoco.mj_forward(m,data);headlocal=data.xmat[head].reshape(3,3).T.copy()
 # EgoDex exported optical camera: +X image right, +Y down, +Z view direction.
 camera_to_forward=np.array([[0,-1,0],[0,0,-1],[1,0,0.]])
 headgoal=g.C@cam[:,:3,:3]@camera_to_forward
 headerrors=[];headviewerrors=[]
 for i in range(n):
  data.qpos[:]=q[i];mujoco.mj_forward(m,data);assert np.max(abs(data.xpos-pp[i]))<1e-5
  previous=q[i-1,headadr] if i else q[i,headadr]
  def head_residual(v):
   data.qpos[headadr]=v;mujoco.mj_forward(m,data)
   actual=data.xmat[head].reshape(3,3)@headlocal
   return np.r_[(actual-headgoal[i]).ravel(),.05*(v-previous)]
  if not head_motion_reference:
   fit=least_squares(head_residual,np.clip(previous,m.jnt_range[headj,0],m.jnt_range[headj,1]),bounds=(m.jnt_range[headj,0],m.jnt_range[headj,1]),max_nfev=80)
   q[i,headadr]=fit.x
  data.qpos[:]=q[i];mujoco.mj_forward(m,data)
  pp[i]=data.xpos;rr[i]=data.xmat.reshape(-1,3,3)
  actual=rr[i,head]@headlocal
  headerrors.append(np.rad2deg(Rotation.from_matrix(actual.T@headgoal[i]).magnitude()))
  headviewerrors.append(np.rad2deg(np.arccos(np.clip(actual[:,0]@headgoal[i,:,0],-1,1))))
 actual_ee=pp[:,ids]+np.einsum('nkij,kj->nki',rr[:,ids],ee_local_np)
 err=np.linalg.norm(actual_ee-raw['position'],axis=-1)*1000
 dr=(rr[:,ids]@raw['semantic_local']).transpose(0,1,3,2)@raw['orientation']
 angle=Rotation.from_matrix(dr.reshape(-1,3,3)).magnitude().reshape(n,2)*180/np.pi
 metrics=g.evaluate(pp,rr,geo);h=pp[:,torso,2]
 np.savez_compressed(folder/'result.npz',qpos=q,body_positions=pp,body_rotations=rr,targets=raw['position'],target_orientation=raw['orientation'],height_release_mask=allowed,held_height_m=h0,arm_reach_ratio=ratio,arm_limit_mask=limit,fixed_height_trial_error_m=e,actual_end_effector_positions=actual_ee,ee_local_position=ee_local_np)
 report=dict(policy=dict(P,source_frames=n,solver='Adam followed by official-bound L-BFGS-B',global_initial_height_solve=global_height,solve_one_constant_setback=solve_setback,penalize_forward_torso_translation=not solve_setback,head_position_reference=head_position_reference,head_position_scale_m=.15 if head_position_reference else None,base_placement='one fixed camera-referenced placement with jointly solved setback; facing yaw preserved' if solve_setback else ('one fixed camera-referenced XY placement; previous facing yaw preserved' if head_placement else P['base_placement'])),frames=n,solved_setback_m=chosen_setback,initialization_objectives=candidate_costs,initial_height_seed_m=height_initial,selected_initialization=list(candidate_costs)[chosen],base_yaw_deg=float(np.rad2deg(anchor[2])),head_orientation_p95_deg=float(np.percentile(headerrors,95)),head_view_direction_p95_deg=float(np.percentile(headviewerrors,95)),base_max_change=float(np.ptp(q[:,:7],axis=0).max()),prior_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),prior_path=str(prior_path),head_position_p95_m=float(np.percentile(np.linalg.norm(pp[:,head]-head_target.cpu().numpy(),axis=-1),95)),
 hand_error_mm_p95=np.percentile(err,95,axis=0).tolist(),orientation_deg_p95=np.percentile(angle,95,axis=0).tolist(),
 torso_tilt_p95_deg=float(np.percentile(metrics['torso_tilt_deg'],95)),torso_tilt_max_deg=float(metrics['torso_tilt_deg'].max()),
 height_reference_m=h0,camera_height_reference_m=camera_height,neutral_head_torso_height_m=head_torso_height,head_height_median_m=float(np.median(pp[:,head,2])),height_range_m=[float(h.min()),float(h.max())],height_release_frames=int(allowed.sum()),
 held_height_error_max_mm=float(np.max(abs(h[~allowed]-h0),initial=0)*1000),minimum_static_com_margin_m=float(metrics['com_margin_m'].min()),
 limitations=['height hold is a strong soft cost, measured deviation reported','reach/limit screen is not a global unreachability proof','static CoM check only','source and robot wrist anatomical conventions remain uncalibrated'])
 report['optimizer_diagnostics']=optimizer_diagnostics
 report['ee_point_definition']=str(raw.get('ee_point_definition','native wrist joint origin'))
 report['ee_local_position_m']=ee_local_np.tolist()
 bend=wrist_bend_angles(pp,rr,ids,proximal,raw['semantic_local'])
 report['wrist_bend_deg_median']=np.median(bend,axis=0).tolist()
 report['wrist_bend_deg_p95']=np.percentile(bend,95,axis=0).tolist()
 report['scene_backoff_m']=scene_backoff_m
 report['policy']['wrist_alignment']='shared soft distal-hand/forearm unit-vector alignment; axial rotation free; 30 degree chord scale, not a hard limit'
 report['policy']['scene_backoff_m']=scene_backoff_m
 report['elapsed_seconds']=time.perf_counter()-started
 report['numerical_initializer']=seed_diagnostics
 report['policy']['solver']='CPU float64 bounded L-BFGS-B; bounded temporal initialization then one polish; windowed stagnation exit'
 report['policy']['penalize_forward_torso_translation']=not head_position_reference
 if initialization_path is not None:
  report['initialization_path']=str(initialization_path)
  report['policy']['initialization']='bounded temporal numerical initialization; original per-frame Prime prior immutable; shared first-frame Prime, neutral, midpoint and Sobol starts'
 if fixed_setback_m is not None:
  report['fixed_setback_m']=fixed_setback_m
  report['policy']['solve_one_constant_setback']=False
 if numerical_resume is not None:
  report['policy']['initialization']='numerical continuation of shared two-start solve; original Prime prior unchanged'
  report['numerical_seed_path']=str(numerical_resume/'result.npz')
 report['policy']['head_motion_reference']=head_motion_reference
 report['policy']['cartesian_continuity_scale_m']=P['position_scale_m'] if head_motion_reference else None
 if head_motion_reference:report['policy']['head_policy']='joint camera rotation and displacement tracking; one constant position offset allowed; no post-solve neck edits'
 head_points=pp[:,head]+(rr[:,head]@head_offset.cpu().numpy()[:,None]).squeeze(-1)
 head_delta=head_points-head_target.cpu().numpy()
 report['policy']['anatomical_head_reference']=anatomical_head_reference
 if anatomical_head_reference:
  report['policy']['head_position_scale_m']=P['position_scale_m']
  report['policy']['head_policy']='joint absolute head-proxy and wrist tracking; declared fixed mesh-derived reference; no free head translation'
  report['policy']['base_placement']='fixed source-camera-referenced placement; yaw faces source wrist workspace'
 if fixed_setback_m is not None:report['policy']['base_placement']='externally specified fixed scene placement; same head, wrist, posture and temporal objective'
 if shoulder_placement:
  report['policy']['base_placement']='source median upper-arm-root midpoint minus neutral robot shoulder-to-chassis offset; optional shared scene backoff recorded separately; XY fixed before solving'
  report['scene_registration']={'source_shoulder_center_m':source_center.tolist(),'neutral_robot_shoulder_offset_m':robot_offset.tolist(),'fixed_base_xy_m':anchor[:2].tolist(),'facing_yaw_unchanged':True}
 report['head_reference_local_position_m']=head_offset.cpu().numpy().tolist()
 report['head_reference_definition']=str(raw.get('head_proxy_definition','head joint origin'))
 report['head_position_p95_m']=float(np.percentile(np.linalg.norm(head_delta,axis=-1),95))
 report['head_height_median_m']=float(np.median(head_points[:,2]))
 report['head_constant_offset_m']=head_delta.mean(0).tolist()
 report['head_motion_error_p95_mm']=float(np.percentile(np.linalg.norm(head_delta-head_delta.mean(0),axis=-1),95)*1000)
 report['head_step_max_mm']=float(np.linalg.norm(np.diff(head_points,axis=0),axis=-1).max()*1000)
 report['torso_step_max_mm']=float(np.linalg.norm(np.diff(pp[:,torso],axis=0),axis=-1).max()*1000)
 report['camera_step_max_mm']=float(np.linalg.norm(np.diff(head_target.cpu().numpy(),axis=0),axis=-1).max()*1000)
 (folder/'report.json').write_text(json.dumps(report,indent=2))
 print(body,json.dumps(report),flush=True)
