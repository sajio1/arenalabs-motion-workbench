"""Vectorized physical FK from global rotations and fixed skeleton metadata."""
import numpy as np
import torch

class GlobalRotationFK(torch.nn.Module):
    def __init__(self,spec,device='cpu',dtype=torch.float32):
        super().__init__()
        parents=np.asarray(spec.parents);count=len(parents)
        paths=np.eye(count,dtype=np.float64)
        for joint,parent in enumerate(parents):
            if parent>=joint:raise ValueError('Skeleton must be in parent-before-child order')
            if parent>=0:paths[joint]+=paths[parent]
        offsets=np.asarray(spec.rest_offsets/100).copy()
        offsets[parents<0]=0  # Translation is a separate world-space variable.
        self.register_buffer('parents',torch.as_tensor(np.maximum(parents,0),device=device,dtype=torch.long))
        self.register_buffer('offsets',torch.as_tensor(offsets,device=device,dtype=dtype))
        self.register_buffer('paths',torch.as_tensor(paths,device=device,dtype=dtype))

    def forward(self,global_rotation):
        bone=(global_rotation.index_select(1,self.parents)@self.offsets[None,:,:,None]).squeeze(-1)
        return torch.matmul(self.paths,bone)
