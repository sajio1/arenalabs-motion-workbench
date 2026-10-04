import numpy as np
import torch
from alphamotion.engine.constraints import _slot_axes
from .global_rotation_fk import GlobalRotationFK
POLICY=dict(name='shared-joint-latent-window-v2-alternating',window_seconds=2.,hop_seconds=1.,steps=200,latent_update_every=5,
    latent_lr=.03,joint_lr=.02,root_position_lr=.005,root_rotation_lr=.01,
    hand_position_scale_m=.005,hand_position_band_m=.002,foot_position_scale_m=.1,
    hand_rotation_scale=.1,foot_rotation_scale=.4,model_rotation_scale=.15,
    model_position_scale_m=.05,model_height_scale_m=.1,
    joint_velocity_scale=.15,joint_acceleration_scale=.03,root_acceleration_scale_m=.005,
    rotation_acceleration_scale=.03,stance_band_m=.03,stance_scale_m=.003,
    support_height_scale_m=.005,penetration_allowance_m=.005,penetration_scale_m=.002,
    support_slip_band_m=.002,support_slip_scale_m=.003)

class PhysicalFK(torch.nn.Module):
    """Depth-batched differentiable FK with official axes and rest rotations."""
    def __init__(self,body,device,dtype=torch.float32):
        super().__init__();spec=body.spec;self.J=spec.J
        parents=np.asarray(spec.parents);self.root=int(np.flatnonzero(parents<0)[0])
        dof=torch.as_tensor(body.dof,device=device,dtype=dtype);axes,active=_slot_axes(dof)
        active[self.root]=False
        self.register_buffer('active_indices',active.flatten().nonzero().flatten())
        self.register_buffer('parents',torch.tensor(np.maximum(parents,0),device=device))
        x,y,z=axes.unbind(-1);zero=torch.zeros_like(x)
        K=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(self.J,3,3,3)
        self.register_buffer('K',K);self.register_buffer('K2',K@K)
        self.register_buffer('identity',torch.eye(3,device=device,dtype=dtype))
        self.register_buffer('rest',torch.as_tensor(body.rest,device=device,dtype=dtype))
        features=dof[:,:18].reshape(self.J,3,6);limited=features[...,3]>.5
        half=features[...,4]*np.pi;center=features[...,5]*np.pi
        self.register_buffer('lower',torch.where(limited,center-half,-torch.inf).flatten()[self.active_indices])
        self.register_buffer('upper',torch.where(limited,center+half,torch.inf).flatten()[self.active_indices])
        depths=[]
        for j,p in enumerate(parents):
            if p>=j:raise ValueError('Non-topological skeleton')
            depths.append(0 if p<0 else depths[p]+1)
        self.level_names=[]
        for depth in range(1,max(depths)+1):
            name=f'level_{depth}';self.register_buffer(name,torch.tensor(np.flatnonzero(np.asarray(depths)==depth),device=device))
            self.level_names.append(name)
        self.positions=GlobalRotationFK(spec,device,dtype)

    def expand_angles(self,active_q):
        return torch.zeros((len(active_q),self.J*3),device=active_q.device,dtype=active_q.dtype).scatter(
            1,self.active_indices[None].expand(len(active_q),-1),active_q).reshape(-1,self.J,3)

    def forward(self,active_q,root_R):
        q=self.expand_angles(active_q)
        elementary=self.identity+q.sin()[...,None,None]*self.K+(1-q.cos())[...,None,None]*self.K2
        local=elementary[:,:,0]@elementary[:,:,1]@elementary[:,:,2]@self.rest
        R=torch.zeros_like(local).index_copy(1,torch.tensor([self.root],device=q.device),root_R[:,None])
        for name in self.level_names:
            ids=getattr(self,name)
            R=R.index_copy(1,ids,R.index_select(1,self.parents[ids])@local.index_select(1,ids))
        return R,self.positions(R)

