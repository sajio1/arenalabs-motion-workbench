"""Bounded numerical continuation; the original per-frame Prime prior is immutable."""
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.stats import qmc
from .wrist_alignment import proximal_origins


def initialize(m, raw, anchor, geo, policy, head, head_offset, head_goal, head_target, n):
    started=time.perf_counter();d=mujoco.MjData(m)
    d.qpos[:]=m.qpos0;d.qpos[:2]=anchor[:2]
    d.qpos[3:7]=[np.cos(anchor[2]/2),0,0,np.sin(anchor[2]/2)]
    mujoco.mj_forward(m,d)
    rest=mujoco.MjData(m);mujoco.mj_forward(m,rest)
    torso=geo['torso'];basis=rest.xmat[torso].reshape(3,3).T
    head_basis=rest.xmat[head].reshape(3,3).T
    ids=raw['wrist_ids'];pi=raw['prior_body_ids'];lo,hi=m.jnt_range[1:].T
    ee_local=raw.get('ee_local_position',np.zeros((len(ids),3)))
    proximal=proximal_origins(m,ids)
    neutral=np.clip(m.qpos0[7:],lo,hi).copy()
    for j in range(1,m.njnt):
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:neutral[m.jnt_qposadr[j]-7]=m.jnt_range[j].mean()
    def residual(x, i):
        d.qpos[7:]=x;mujoco.mj_forward(m,d);r=d.xmat.reshape(-1,3,3);p=d.xpos
        ee=p[ids]+np.einsum('kij,kj->ki',r[ids],ee_local)
        parts=[((ee-raw['position'][i])/(policy['position_scale_m']*np.sqrt(6))).ravel(),
               ((r[ids]@raw['semantic_local']-raw['orientation'][i])/(policy['orientation_scale']*np.sqrt(18))).ravel(),
               ((r[pi]-raw['prior_R'][i])*np.sqrt(policy['prior_rotation_weight']/(len(pi)*9))).ravel(),
               (p[head]+r[head]@head_offset-head_target[i])/(policy['position_scale_m']*np.sqrt(3)),
               ((r[head]@head_basis-head_goal[i])/(policy['orientation_scale']*3)).ravel()]
        forearm=p[ids]-p[proximal]
        forearm/=np.maximum(np.linalg.norm(forearm,axis=-1,keepdims=True),1e-12)
        parts.append(((forearm-(r[ids]@raw['semantic_local'])[...,0])/(policy['wrist_alignment_scale_rad']*np.sqrt(len(ids)))).ravel())
        rel=r[1].T@(p[torso]-p[1]);rr=r[1].T@r[torso]@basis
        forward=np.arctan2(rr[0,2],rr[2,2]);lateral=np.arctan2(rr[1,2],rr[2,2]);yaw=np.arctan2(rr[1,0],rr[0,0])
        excess=max(forward-np.deg2rad(policy['torso_tilt_free_deg']),0)+max(-forward,0)
        parts.extend([np.array([(rel[1]-geo['torso_xy_rest'][1])/policy['torso_horizontal_scale_m']])*np.sqrt(policy['posture_weight']),
                      np.array([excess,lateral,yaw])*np.sqrt(policy['posture_weight'])/policy['torso_tilt_scale']])
        cm=((p+np.einsum('bij,bj->bi',r,geo['body_ipos']))*geo['mass'][:,None]).sum(0)
        local=r[1].T@(cm-p[1]);eq=geo['hull']
        edge=np.maximum(local[:2]@eq[:,:2].T+eq[:,2]+geo['radius']*policy['com_margin_fraction'],0)/(geo['radius']*policy['com_edge_scale_fraction'])
        parts.append(edge/np.sqrt(len(edge)))
        return np.concatenate(parts)
    # Explore coordinate branches on ONE frame, rather than re-solving the
    # entire sequence for every start. Sampling depends only on official
    # bounds and DOF count, never embodiment identity or action labels.
    samples=qmc.Sobol(d=len(lo),scramble=True,seed=0).random_base2(3)
    starts=[np.clip(raw['seed'][0,7:],lo,hi),neutral,(lo+hi)/2,*list(lo+samples*(hi-lo))]
    fits=[least_squares(residual,v,args=(0,),bounds=(lo,hi),max_nfev=100,ftol=1e-6,xtol=1e-6,gtol=1e-6) for v in starts]
    best=min(fits,key=lambda f:np.sum(f.fun**2));q=[best.x]
    evaluations=sum(f.nfev for f in fits)
    for i in range(1,n):
        fit=least_squares(residual,q[-1],args=(i,),bounds=(lo,hi),max_nfev=25,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        q.append(fit.x);evaluations+=fit.nfev
    return np.asarray(q),dict(seconds=time.perf_counter()-started,evaluations=evaluations,first_frame_candidate_costs=[float(np.sum(f.fun**2)) for f in fits],
        definition='shared first-frame Prime, neutral, midpoint and eight Sobol starts within official bounds; bounded continuation only initializes; immutable per-frame Prime prior')
