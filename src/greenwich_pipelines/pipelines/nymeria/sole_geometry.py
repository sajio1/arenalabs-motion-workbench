"""Fixed sole contact points measured from vendor visual mesh geometry."""
import numpy as np
import torch
import mujoco as mj
from scipy.spatial import ConvexHull
from alphamotion.viz.kinematics import visual_mesh_geom_ids
from alphamotion.engine.constraints import fk_from_angles
C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])

def sole_surface(body,indices):
    model=mj.MjModel.from_xml_path(body.xml);data=mj.MjData(model)
    data.qpos[:]=model.qpos0;mj.mj_forward(model,data)
    R,P=fk_from_angles(torch.zeros((1,body.spec.J,3),dtype=torch.float64),body.spec,
        torch.tensor(body.dof,dtype=torch.float64),rest=torch.tensor(body.rest,dtype=torch.float64),
        root_R=torch.eye(3,dtype=torch.float64)[None])
    result=[];hulls=[]
    for index in indices:
        bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,body.spec.joint_names[index])
        vertices=[]
        for gid in visual_mesh_geom_ids(model):
            ancestor=int(model.geom_bodyid[gid])
            while ancestor>0 and ancestor!=bid:ancestor=int(model.body_parentid[ancestor])
            if ancestor!=bid:continue
            mid=int(model.geom_dataid[gid]);start=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid])
            vertices.append(model.mesh_vert[start:start+count]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid])
        if not vertices:raise ValueError('No vendor foot mesh for '+body.spec.joint_names[index])
        v=np.concatenate(vertices);bottom=v[v[:,2]<=v[:,2].min()+.002]
        # MuJoCo X is forward. Two actual sole vertices for the heel and
        # two for the toe make the recorded heel/toe labels unambiguous.
        span=np.ptp(bottom[:,0])
        heel=bottom[bottom[:,0]<=bottom[:,0].min()+.2*span]
        toe=bottom[bottom[:,0]>=bottom[:,0].max()-.2*span]
        chosen=np.stack([heel[np.argmin(heel[:,1])],heel[np.argmax(heel[:,1])],
                         toe[np.argmin(toe[:,1])],toe[np.argmax(toe[:,1])]])
        points=(chosen-data.xpos[bid])@C.T
        result.append(points@R[0,index].numpy())
        # Any lowest vertex under a rigid rotation lies on the convex hull.
        # Preserve the whole foot envelope, not only its neutral bottom band.
        hull=v[ConvexHull(v).vertices]
        hulls.append(((hull-data.xpos[bid])@C.T)@R[0,index].numpy())
    length=max(len(hull) for hull in hulls)
    padded=np.zeros((2,length,3));valid=np.zeros((2,length),bool)
    for foot,hull in enumerate(hulls):
        padded[foot,:len(hull)]=hull;valid[foot,:len(hull)]=True
    return np.asarray(result),padded,valid

def sole_points(body,indices):
    return sole_surface(body,indices)[0]
