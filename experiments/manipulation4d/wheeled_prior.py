"""SMPL-X-conditioned mobile manipulation with immutable world-space goals.

Uses native mixed hinge/slide FK. A planar chassis is optimized jointly with the
torso and arms. Wheel actuation is not inferred from fixed visual wheel links.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch,mujoco
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from greenwich_pipelines.pipelines.gento import source as g
from greenwich_pipelines.runtime import load_model,configure
from support_reference import support_reference


POLICY='smplx-six-encode-fixed-base-light-refine-v5'

def prepare(row,bodywork,out,targets,desired,offsets):
    configure('/opt/arenalab/configs/runtime.cloud.json')
    torch.set_num_threads(2)
    bodywork=Path(bodywork);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads(Path(row['metadata']).read_text());model=mujoco.MjModel.from_xml_path(row['xml'])
    if model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE:raise ValueError('Planar chassis requires a free root in native metadata')
    inputs={name:hashlib.sha256((bodywork/name).read_bytes()).hexdigest() for name in ['canonical.npz','smplx_motion.npz','observations.npz','operation-accuracy/source-targets.npz']}
    human=np.load(bodywork/'smplx_motion.npz');n=len(targets)
    floor,_,_,floor_report=support_reference(human,np.load(bodywork/'SMPLX_NEUTRAL.npz',allow_pickle=True))
    base_z=float(floor+meta['ground_offset'])
    ids=[model.body(w).id for w in meta['wrists']];torso=model.body(meta['torso']).id
    emb=g.load_from_xml(row['xml'],row['body']);spec=emb.spec;keep,frame=g.build_merge(model)
    wi=[next(i for i,b in enumerate(keep) if frame[b]==bid or b==bid) for bid in ids]
    ti=next(i for i,b in enumerate(keep) if frame[b]==torso or b==torso)
    # SMPL-X anatomical +Z faces forward, +Y points up in its local frame.
    basis=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    pelvisR=human['global_rotation'][:,0]@basis.T
    yaw=np.unwrap(np.arctan2(pelvisR[:,1,0],pelvisR[:,0,0]))
    anchor=np.c_[human['joints'][:,0,:2],yaw]
    gauge=np.c_[anchor[:,:2],np.full(n,base_z)]
    rest=mujoco.MjData(model);mujoco.mj_forward(model,rest)
    torsoR=human['global_rotation'][:,9]@basis.T@rest.xmat[torso].reshape(3,3)
    native_targets=targets-np.einsum('tsij,tsj->tsi',desired,offsets)
    reach=np.linalg.norm(g.rest_positions(spec),axis=1).max()
    baseR=Rotation.from_euler('z',yaw).as_matrix()
    inv=baseR.transpose(0,2,1)
    from greenwich_motion_sdk.motion import MotionClip
    from alphamotion.engine.nets.rotations import SkeletonSpec
    from alphamotion.embodiment.dof_features import ball_dof
    from alphamotion.engine.spatial import fk_pos
    clip=MotionClip.load(bodywork/'canonical.npz')
    human_spec=SkeletonSpec('sdk_smplx',clip.parents,clip.rest_offsets_cm.astype(np.float32),clip.names)
    hd,_=ball_dof(len(clip.names))
    gr=clip.global_rotation;gp=clip.world_position_cm-clip.world_position_cm[:,:1]
    r6=g.matrix_to_rot6d(torch.tensor(gr,dtype=torch.float32))
    fk_error=float(np.linalg.norm(fk_pos(r6.numpy(),human_spec)-gp,axis=-1).max())
    if fk_error>.001:raise ValueError('Source SMPL FK mismatch')
    pose=np.concatenate([r6.numpy(),gp/float(np.linalg.norm(gp,axis=-1).max())],-1).astype(np.float32)
    from greenwich_pipelines.input_policy import encode_observed, receipt
    role_indices=[clip.names.index(name) for name in ['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']]
    input_policy=receipt(clip.names,role_indices)
    raw=out/'prime_original.npz'
    if raw.exists() and str(np.load(raw).get('encoder_policy','legacy-full'))!=input_policy['policy']:
        raw.rename(out/('prime_original.legacy-'+str(time.time_ns())+'.npz'))
    if raw.exists():
        prior=dict(np.load(raw));assert np.array_equal(prior['input_pose9'],pose)
    else:
        prime=load_model()
        with torch.no_grad():
            codes=torch.cat([encode_observed(prime,torch.tensor(pose[k:k+128],device=prime.device),human_spec,hd,role_indices) for k in range(0,n,128)])
            rot,pos=prime.decode_full(codes,spec,emb.dof)
            _,_,q=g.project(rot,spec,torch.tensor(emb.dof,device=prime.device),rest=torch.tensor(emb.rest,device=prime.device))
        seed=np.broadcast_to(model.qpos0[7:],(n,model.nq-7)).copy()
        for j,names in enumerate(emb.qnames):
            for k,name in enumerate(names):
                ji=model.joint(name).id
                if model.jnt_type[ji]==mujoco.mjtJoint.mjJNT_HINGE:seed[:,model.jnt_qposadr[ji]-7]=q[:,j,k].cpu().numpy()
        prior=dict(encoder_policy=np.array(input_policy['policy']),input_pose9=pose,raw_rot6d=rot.cpu().numpy(),raw_pos=pos.cpu().numpy(),codes=codes.cpu().numpy(),seed=seed,
            prior_R=g.AX.T@g.rot6d_to_matrix(rot).cpu().numpy()@g.AX,prior_body_ids=np.array([frame[b] for b in keep]))
        np.savez_compressed(raw,**prior);del prime;torch.cuda.empty_cache()
    (out/'input-mapping.json').write_text(json.dumps(dict(**input_policy,source_names=clip.names,fk_error_cm=fk_error),indent=2))
    raw_hash=hashlib.sha256(raw.read_bytes()).hexdigest()
    return raw
