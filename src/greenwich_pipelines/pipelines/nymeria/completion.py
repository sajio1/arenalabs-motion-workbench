"""Production four-endpoint target-body completion, with immutable raw output."""
import numpy as np
import torch
from alphamotion.engine.descriptor import geom_batch
from alphamotion.engine.spatial import rest_positions
from alphamotion.engine.constraints import project
from alphamotion.engine.nets.rotations import matrix_to_rot6d
from .input import endpoint_indices
from .orientation import robot_semantic_local

def complete(prime,body,robot,source):
    p=source['position_yup_cm'];spec,dof=body.spec,body.dof
    root=int(np.flatnonzero(np.asarray(spec.parents)<0)[0]);ids=endpoint_indices(spec,robot)
    desired=source['semantic_rotation_yup']@robot_semantic_local(robot,ids).transpose(0,2,1)[None]
    r6=matrix_to_rot6d(torch.tensor(desired)).numpy()
    gauge=p[:,2:].mean(1);gauge[:,1]=spec.rest_offsets[root,1]
    reach=np.linalg.norm(rest_positions(spec),axis=1).max()
    pose=np.zeros((len(p),spec.J,9),np.float32);pose[...,:6]=[1,0,0,0,1,0]
    pose[:,ids,:6]=r6;pose[:,ids,6:]=(p-gauge[:,None])/reach
    hidden=torch.ones((len(p),spec.J),device=prime.device,dtype=torch.bool);hidden[:,ids]=False
    prime._geoms.pop(spec.name,None)
    with torch.no_grad():
        features=[]
        for start in range(0,len(p),256):
            features.append(prime.model.encode_features(torch.tensor(pose[start:start+256],device=prime.device),geom_batch(prime.geom(spec,dof),len(pose[start:start+256])),joint_mask=hidden[start:start+256]))
        fp=torch.cat([x[0] for x in features]);fr=torch.cat([x[1] for x in features])
        ip,ir=prime.model.ints_from_features(fp,fr);codes=torch.cat((ip[0],ir[0]),1).long()
        rot,pos=prime.decode_full(codes,spec,dof)
        projected,local,q=project(rot,spec,torch.tensor(dof,device=prime.device),rest=torch.tensor(body.rest,device=prime.device))
        local=local.cpu().numpy();t=np.zeros_like(gauge);t[:,1]=pos[:,root,0].cpu().numpy()*100
        t[:,[0,2]]=(p[:,2:].mean(1)-local[:,ids[2:]].mean(1))[:,[0,2]]
    return dict(q=q.cpu().numpy(),rot6d=projected.cpu().numpy(),raw_rot6d=rot.cpu().numpy(),raw_pos=pos.cpu().numpy(),root_t=t,joints_yup_cm=local+t[:,None],input_positions_yup_cm=p,input_rot6d=r6,target_rotation_yup=desired,endpoint_indices=np.asarray(ids),parents=spec.parents,joint_names=np.asarray(spec.joint_names),input_gauge_cm=gauge,original_features_p=fp.cpu().numpy(),original_features_r=fr.cpu().numpy(),codes=codes.cpu().numpy(),pose9=pose,fps=np.array(30.))
