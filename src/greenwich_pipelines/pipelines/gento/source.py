import numpy as np
import torch,mujoco,h5py
from pathlib import Path
from scipy.spatial.transform import Rotation
from alphamotion.embodiment.registry import load_from_xml
from alphamotion.embodiment.mjcf_build import AX,build_merge
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import geom_batch
from alphamotion.engine.spatial import rest_positions
from alphamotion.engine.constraints import project
from alphamotion.engine.nets.rotations import matrix_to_rot6d,rot6d_to_matrix
from .posture_geometry import geometry,evaluate
ROOT=None
C=np.array([[0,0,-1],[-1,0,0],[0,1,0.]])
POLICY=dict(fps=30,source_frames=450)


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def source(path):
    with h5py.File(path) as h:
        n=min(POLICY['source_frames'],len(h['transforms/hip'])); tr={k:np.array(v[:n]) for k,v in h['transforms'].items()}
        p=np.stack([tr[s+'Hand'][:,:3,3]@C.T for s in ['left','right']],1)
        frames=[]
        for side,sign in [('left',1),('right',-1)]:
            f=unit(tr[side+'MiddleFingerKnuckle'][:,:3,3]-tr[side+'Hand'][:,:3,3])
            lateral=tr[side+'IndexFingerKnuckle'][:,:3,3]-tr[side+'LittleFingerKnuckle'][:,:3,3]
            normal=unit(np.cross(f,lateral)*sign); y=unit(np.cross(normal,f))
            frames.append(C@np.stack([f,y,normal],-1))
        return p,np.stack(frames,1),tr['hip'][:,:3,3]@C.T


def metadata(model,meta):
    # Fixed physical mounting definitions are part of the embodiment metadata.
    # Solvers do not infer hand mounting from the current forearm pose.
    frames=meta.get('end_effector_frames')
    if frames is None:raise ValueError('Missing audited end-effector mounting frames')
    by_link={frame['link']:frame for frame in frames}
    ids=[model.body(w).id for w in meta['wrists']]
    local=np.array([by_link[w]['rotation'] for w in meta['wrists']])
    if not np.allclose(local.transpose(0,2,1)@local,np.eye(3),atol=1e-8):raise ValueError('Invalid mounting rotation')
    if not np.allclose(np.linalg.det(local),1.,atol=1e-8):raise ValueError('Mounting frame must be right handed')
    return ids,local


class FK:
    def __init__(self,m,device,anchor,dtype=torch.float32):
        self.m=m;self.device=device;self.t=lambda v:torch.tensor(v,dtype=dtype,device=device)
        self.pos=self.t(m.body_pos);self.rot=self.t(Rotation.from_quat(m.body_quat[:,[1,2,3,0]]).as_matrix())
        self.axis=self.t(m.jnt_axis);self.anchor=self.t(m.jnt_pos)
        self.base_anchor=self.t(anchor)
    def __call__(self,x):
        N=len(x);I=torch.eye(3,device=x.device,dtype=x.dtype).expand(N,3,3);zero=torch.zeros((N,3),device=x.device,dtype=x.dtype)
        ps=[zero];rs=[I]
        for b in range(1,self.m.nbody):
            par=int(self.m.body_parentid[b]);r=rs[par]@self.rot[b];p=ps[par]+(rs[par]@self.pos[b,:,None]).squeeze(-1)
            if b==1:
                a=self.base_anchor[2];c=a.cos();s=a.sin();zz=a*0;one=zz+1
                r=torch.stack([c,-s,zz,s,c,zz,zz,zz,one]).reshape(3,3).expand(N,3,3)
                p=torch.stack([self.base_anchor[0],self.base_anchor[1],zz+self.m.qpos0[2]]).expand(N,3)
            else:
                for j in range(self.m.body_jntadr[b],self.m.body_jntadr[b]+self.m.body_jntnum[b]):
                    a=x[:,self.m.jnt_qposadr[j]-7];axis=self.axis[j]
                    if self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:
                        p=p+(r@axis[:,None]).squeeze(-1)*a[:,None]
                    elif self.m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:
                        vx,vy,vz=axis;z=vx*0;K=torch.stack([z,-vz,vy,vz,z,-vx,-vy,vx,z]).reshape(3,3)
                        dr=I+a.sin()[:,None,None]*K+(1-a.cos())[:,None,None]*(K@K)
                        newr=r@dr;p=p+(r@self.anchor[j,:,None]).squeeze(-1)-(newr@self.anchor[j,:,None]).squeeze(-1);r=newr
            ps.append(p);rs.append(r)
        return torch.stack(ps,1),torch.stack(rs,1)

