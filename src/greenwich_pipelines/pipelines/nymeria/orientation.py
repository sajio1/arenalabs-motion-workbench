import numpy as np
import torch
from alphamotion.engine import constraints
from greenwich_umi_proof.palm_frames import palm_frame,robot_palm_local
from .embodiment_exact import load_robot
C=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
SOURCE_JOINTS=(14,10,21,17)
SOURCE_PARENTS=(13,9,22,18)


def source_semantic_frames(tpose,relatives):
    tpose=np.asarray(tpose).reshape(23,3)
    relatives=np.asarray(relatives,dtype=np.float64)
    bases=[]
    for i,(joint,other) in enumerate(zip(SOURCE_JOINTS,SOURCE_PARENTS)):
        distal=(tpose[joint]-tpose[other]) if i<2 else (tpose[other]-tpose[joint])
        # MVN manual MV0319P.N section 8.2.3: arms horizontal, thumbs forward.
        # Both palmar normals therefore point DOWN in the Z-up T-pose.
        # Segment-frame +X is a reference axis, not a hand-back normal.
        # This is fixed dataset anatomy, shared by all motions and robots.
        normal=np.array([0.,0.,-1.]) if i<2 else np.array([0.,0.,1.])
        if i>=2:
            # An ankle-to-toe bone slopes down even in neutral standing.
            # Foot forward is its projection into the neutral sole plane;
            # preserve the declared up normal rather than tilting it to
            # become perpendicular to the ankle-to-toe bone.
            distal=distal-normal*np.dot(distal,normal)
        bases.append(C@palm_frame(distal,normal))
    return relatives@np.asarray(bases)[None]


def robot_semantic_local(robot,indices):
    target=load_robot(robot)
    spec=target.spec
    dtype=torch.float64
    R,p=constraints.fk_from_angles(
        torch.zeros((1,len(spec.parents),3),dtype=dtype),spec,
        torch.as_tensor(target.dof,dtype=dtype),
        rest=torch.as_tensor(target.rest,dtype=dtype),
        root_R=torch.eye(3,dtype=dtype)[None])
    R=R[0].numpy();p=p[0].numpy()
    frames=[]
    for i,side in enumerate(('left','right','left','right')):
        if i<2:
            frames.append(robot_palm_local(robot['palm_frames'],f'{side}_wrist'))
            continue
        else:
            distal=np.array([0.,0.,1.]);normal=np.array([0.,1.,0.])
        frames.append(R[indices[i]].T@palm_frame(distal,normal))
    return np.asarray(frames)


def angular_errors_deg(desired,achieved):
    cos=(np.einsum('...ij,...ij->...',desired,achieved)-1.)/2.
    return np.rad2deg(np.arccos(np.clip(cos,-1.,1.)))

