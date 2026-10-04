"""V3 candidate: physical joint/root solve from immutable raw Prime output.

No action/robot policy branches, previous refined seeds, mesh rotations, or
prescribed pelvis height. Shared soft trunk costs guide redundant posture.
Model output remains a frozen reference rather than an optimized target.
"""
import json
from time import perf_counter
import numpy as np
import torch
from scipy.optimize import minimize
from .physical_fk import PhysicalFK, POLICY as V2_POLICY
from .sole_geometry import sole_surface
from alphamotion.engine.nets.rotations import rot6d_to_matrix, matrix_to_rot6d

POLICY=dict(V2_POLICY, name='shared-v3-posture-quadratic',
    optimizer='L-BFGS-B-official-joint-bounds',max_iterations=1200,history_size=30,
    trunk_gravity_scale=.25,trunk_lower_hemisphere_scale=.1,
    trunk_direction_velocity_scale=.05,
    model_height_policy='predicted height initializes root; no independent height spring',
    initialization='immutable raw Prime physical projection',model_reference='immutable raw rotations',
    visual_binding='official mesh, no dummy correction')

def fit(prime,body,reference,seed,contacts,fps):
    # The generic evaluation harness supplies a legacy seed; V3 deliberately
    # initializes every evaluation from the original model reference instead.
    start_time=perf_counter();device=prime.device
    cast=lambda x:torch.as_tensor(x,device=device,dtype=torch.float64)
    fk=PhysicalFK(body,device,torch.float64);root=fk.root
    ids=np.asarray(reference['endpoint_indices']).tolist()
    target=cast(reference['input_positions_yup_cm']/100)
    desired=cast(reference['target_rotation_yup'])
    prior=rot6d_to_matrix(cast(reference.get('raw_rot6d',reference['rot6d']))).detach()
    q0=cast(reference['q']).flatten(1)[:,fk.active_indices]
    u=q0.clamp(fk.lower,fk.upper).detach().clone().requires_grad_()
    g=cast(reference['root_t']/100).clone().requires_grad_()
    r=cast(reference['rot6d'][:,root]).clone().requires_grad_()
    from greenwich_pipelines.robot_semantics import shoulder_landmarks
    shoulders=shoulder_landmarks(body.spec,ids[:2])
    sole,hull,valid=sole_surface(body,ids[2:]);sole=cast(sole);hull=cast(hull)
    valid=torch.as_tensor(valid,device=device);flags=cast(contacts)
    scales=cast([POLICY['hand_rotation_scale']]*2+[POLICY['foot_rotation_scale']]*2)
    target_d2=desired[2:]-2*desired[1:-1]+desired[:-2]
    cadence=fps/30.;history=[];evaluations=0
    def evaluate():
        q=u
        R,relative=fk(q,rot6d_to_matrix(r));world=relative+g[:,None]
        error=world[:,ids]-target;distance=error[:,:2].norm(dim=-1)
        terms={}
        terms['hand_position']=(torch.relu(distance-POLICY['hand_position_band_m'])/POLICY['hand_position_scale_m']).square().mean()
        terms['foot_position']=(error[:,2:]/POLICY['foot_position_scale_m']).square().mean()
        delta=R[:,ids]@desired.transpose(-1,-2)
        skew=torch.stack((delta[...,2,1]-delta[...,1,2],delta[...,0,2]-delta[...,2,0],delta[...,1,0]-delta[...,0,1]),-1)*.5
        angle=torch.atan2(skew.norm(dim=-1),.5*(delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1))
        terms['orientation']=(2/9*(angle/scales).square()).mean()
        terms['model_rotation']=((R-prior)/POLICY['model_rotation_scale']).square().mean()
        terms['model_position']=((relative-fk.positions(prior))/POLICY['model_position_scale_m']).square().mean()
        sw=torch.einsum('tfij,fkj->tfki',R[:,ids[2:]],sole)+world[:,ids[2:],None]
        hw=torch.einsum('tfij,fkj->tfki',R[:,ids[2:]],hull)+world[:,ids[2:],None]
        terms['support_height']=(flags*(sw[...,1]/POLICY['support_height_scale_m']).square()).mean()
        terms['penetration']=(torch.relu(-POLICY['penetration_allowance_m']-hw[...,1])[:,valid]/POLICY['penetration_scale_m']).square().mean()
        horizontal=error[:,2:,[0,2]]
        vectors=torch.cat((horizontal,(horizontal[:,1]-horizontal[:,0])[:,None]),1)
        terms['stance']=(torch.relu(vectors.norm(dim=-1)-POLICY['stance_band_m'])/POLICY['stance_scale_m']).square().mean()
        trunk=world[:,shoulders].mean(1)-world[:,root]
        up=trunk[:,1]/trunk.norm(dim=-1).clamp_min(1e-9)
        # Squared chord distance is quadratic near upright; the former squared
        # cosine deficit was quartic and barely penalized moderate lean.
        terms['trunk_gravity']=(2*(1-up)/POLICY['trunk_gravity_scale']**2).mean()
        trunk_direction=trunk/trunk.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        # Penalize torso-direction motion explicitly instead of diluting it
        # across all joints. This leaves root translation and gait unconstrained.
        terms['trunk_direction_velocity']=(((trunk_direction[1:]-trunk_direction[:-1])*cadence/POLICY['trunk_direction_velocity_scale']).square().sum(-1)).mean()
        terms['trunk_lower_hemisphere']=(torch.relu(-up)/POLICY['trunk_lower_hemisphere_scale']).square().mean()
        terms['joint_velocity']=((q[1:]-q[:-1])*cadence/POLICY['joint_velocity_scale']).square().mean()
        terms['joint_acceleration']=((q[2:]-2*q[1:-1]+q[:-2])*cadence**2/POLICY['joint_acceleration_scale']).square().mean()
        terms['root_acceleration']=((g[2:]-2*g[1:-1]+g[:-2])*cadence**2/POLICY['root_acceleration_scale_m']).square().mean()
        d2=R[2:]-2*R[1:-1]+R[:-2]
        desired_all=torch.zeros_like(d2);desired_all[:,ids]=target_d2
        terms['rotation_acceleration']=((d2-desired_all)*cadence**2/POLICY['rotation_acceleration_scale']).square().mean()
        slip=(sw[1:]-sw[:-1])[...,[0,2]].norm(dim=-1)
        terms['slip']=(torch.minimum(flags[1:],flags[:-1])*(torch.relu(slip-POLICY['support_slip_band_m'])/POLICY['support_slip_scale_m']).square()).mean()
        return sum(terms.values()),terms,q,R,world,sw,up
    parameters=[u,g,r];sizes=[v.numel() for v in parameters]
    def closure(x):
        nonlocal evaluations
        offset=0
        with torch.no_grad():
            for v,size in zip(parameters,sizes):
                v.copy_(cast(x[offset:offset+size]).reshape(v.shape));v.grad=None;offset+=size
        loss,terms,*_=evaluate()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite V3 objective')
        loss.backward();evaluations+=1
        if evaluations==1 or evaluations%50==0:
            row=dict(evaluation=evaluations,objective=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()})
            history.append(row);print('V3',json.dumps(row),flush=True)
        gradient=np.concatenate([v.grad.detach().cpu().numpy().ravel() for v in parameters])
        return float(loss.detach()),gradient
    x0=np.concatenate([v.detach().cpu().numpy().ravel() for v in parameters])
    lower=np.concatenate([np.tile(fk.lower.cpu().numpy(),len(target)),np.full(sizes[1]+sizes[2],-np.inf)])
    upper=np.concatenate([np.tile(fk.upper.cpu().numpy(),len(target)),np.full(sizes[1]+sizes[2],np.inf)])
    optimized=minimize(closure,x0,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),
      options=dict(maxiter=POLICY['max_iterations'],maxcor=POLICY['history_size'],ftol=1e-10,gtol=1e-7,maxls=30))
    closure(optimized.x)
    with torch.no_grad():
        loss,terms,q,R,world,sw,up=evaluate()
        result=dict(q=fk.expand_angles(q).cpu().numpy(),rot6d=matrix_to_rot6d(R).cpu().numpy(),
          root_t=g.cpu().numpy()*100,joints_yup_cm=world.cpu().numpy()*100,
          sole_world_yup_m=sw.cpu().numpy(),contact_point_mask=np.asarray(contacts,dtype=bool),
          endpoint_indices=ids,parents=body.spec.parents,joint_names=body.spec.joint_names,
          input_positions_yup_cm=reference['input_positions_yup_cm'],target_rotation_yup=reference['target_rotation_yup'],
          initialization_policy=POLICY['initialization'],external_seed_used=False)
    return result,dict(seconds=perf_counter()-start_time,closure_evaluations=evaluations,history=history,
      optimizer_iterations=int(optimized.nit),iteration_limit_reached=int(optimized.nit)>=POLICY['max_iterations'],
      optimizer_success=bool(optimized.success),optimizer_message=str(optimized.message),
      final_terms={k:float(v) for k,v in terms.items()},trunk_up_cosine_min=float(up.min()),
      initialization_policy=POLICY['initialization'],raw_model_prior_immutable=True,accepted=False)
