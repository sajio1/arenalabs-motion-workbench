import hashlib,json
from pathlib import Path
import h5py,mujoco,numpy as np,torch
from . import source as g
from .head_reference import reference
ROOT=None
RAW_VERSION='raw'
FRAMES=None


def prepare(body,clip,target_forward_m=0.,source_path=None,frames=FRAMES,require_video=True):
    if source_path is None:raise ValueError('Explicit source_path required')
    path=Path(source_path)
    with h5py.File(path) as h:
        total=len(h['transforms/camera']);n=total if frames is None else int(frames)
        assert 3<=n<=total
        cam=np.asarray(h['transforms/camera'][:n])
    meta=json.loads((ROOT/'models'/body/'metadata.json').read_text())
    ee_local=np.array([next(f['origin_m'] for f in meta['end_effector_frames'] if f['link']==w) for w in meta['wrists']])
    folder=ROOT/RAW_VERSION/body/clip;folder.mkdir(parents=True,exist_ok=True)
    output=folder/'prime_original.npz'
    if output.exists():
        with np.load(output) as existing:
            existing_shift=float(existing['target_forward_m']) if 'target_forward_m' in existing else 0.
            assert abs(existing_shift-target_forward_m)<1e-12
            assert np.allclose(existing['ee_local_position'],ee_local,atol=1e-10,rtol=0), 'Existing prior uses another EE frame; create a new raw version, never overwrite it'
            assert len(existing['position'])==n and Path(str(existing['source'])).resolve()==path.resolve()
        return output
    if require_video:assert path.with_suffix('.mp4').exists()
    g.POLICY['source_frames']=n
    p,R,hip=g.source(path)
    assert len(p)==n and len(R)==n and len(hip)==n
    m=mujoco.MjModel.from_xml_path(meta['model']);emb=g.load_from_xml(meta['model'],body)
    spec=emb.spec;keep,frame=g.build_merge(m);ids,semantic=g.metadata(m,meta)
    wrists=[next(i for i,b in enumerate(keep) if frame[b]==bid or b==bid) for bid in ids]
    head,head_offset,_=reference(m,meta,face_surface=True)
    hn=next(i for i,b in enumerate(keep) if frame[b]==head or b==head)
    rest=mujoco.MjData(m);mujoco.mj_forward(m,rest)
    camera_to_forward=np.array([[0,-1,0],[0,0,-1],[1,0,0.]])
    head_R=g.C@cam[:,:3,:3]@camera_to_forward@rest.xmat[head].reshape(3,3)
    camera_p=cam[:,:3,3]@g.C.T
    head_p=camera_p-(head_R@head_offset[:,None]).squeeze(-1)
    original_position=p.copy()
    forward=np.median(p.mean(1),axis=0)[:2]-np.median(camera_p,axis=0)[:2]
    forward=forward/max(np.linalg.norm(forward),1e-12)
    task_shift=np.r_[forward*float(target_forward_m),0.]
    p=p+task_shift
    gauge=hip.copy();gauge[:,2]=meta['ground_offset']
    reach=np.linalg.norm(g.rest_positions(spec),axis=1).max()
    pose=np.zeros((n,spec.J,9),np.float32);pose[...,:6]=[1,0,0,0,1,0]
    desired=R@semantic.transpose(0,2,1)[None]
    # Prime nodes remain the native joint origins. Invert the fixed tool
    # transform using the observed orientation before conditioning the model.
    native_wrist_target=p-np.einsum('nkij,kj->nki',desired,ee_local)
    pose[:,wrists,:6]=g.matrix_to_rot6d(torch.tensor(g.AX@desired@g.AX.T)).numpy()
    pose[:,wrists,6:]=(native_wrist_target-gauge[:,None])@g.AX.T*100/reach
    pose[:,hn,:6]=g.matrix_to_rot6d(torch.tensor(g.AX@head_R@g.AX.T)).numpy()
    pose[:,hn,6:]=(head_p-gauge)@g.AX.T*100/reach
    assert np.isfinite(pose).all()
    from greenwich_pipelines.runtime import load_model
    prime=load_model()
    hidden=torch.ones((n,spec.J),device=prime.device,dtype=torch.bool)
    hidden[:,wrists+[hn]]=False
    with torch.no_grad():
        feats=[]
        for start in range(0,n,128):
            z=torch.tensor(pose[start:start+128],device=prime.device)
            feats.append(prime.model.encode_features(z,g.geom_batch(prime.geom(spec,emb.dof),len(z)),joint_mask=hidden[start:start+128]))
        fp=torch.cat([v[0] for v in feats]);fr=torch.cat([v[1] for v in feats])
        ip,ir=prime.model.ints_from_features(fp,fr);codes=torch.cat((ip[0],ir[0]),1).long()
        rot,pos=prime.decode_full(codes,spec,emb.dof)
        _,_,q=g.project(rot,spec,torch.tensor(emb.dof,device=prime.device),rest=torch.tensor(emb.rest,device=prime.device))
    seed=np.broadcast_to(m.qpos0,(n,m.nq)).copy();seed[:,:2]=hip[:,:2]
    for i,names in enumerate(emb.qnames):
        for k,name in enumerate(names):
            j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
            if j>=0 and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:seed[:,m.jnt_qposadr[j]]=q[:,i,k].cpu().numpy()
    np.savez_compressed(output,raw_rot6d=rot.cpu().numpy(),raw_pos=pos.cpu().numpy(),codes=codes.cpu().numpy(),
        features_p=fp.cpu().numpy(),features_r=fr.cpu().numpy(),seed=seed,
        prior_R=g.AX.T@g.rot6d_to_matrix(rot).cpu().numpy()@g.AX,prior_body_ids=np.array([frame[b] for b in keep]),
        position=p,orientation=R,semantic_local=semantic,wrist_ids=ids,hip=hip,input_pose9=pose,
        original_position=original_position,target_forward_m=float(target_forward_m),task_translation_world_m=task_shift,
        source=str(path),joint_names=spec.joint_names,head_position=head_p,head_orientation=head_R,
        observed_nodes=np.array(wrists+[hn]),head_body_id=head,head_reference_local_position=head_offset,
        ee_local_position=ee_local,native_wrist_target=native_wrist_target,
        ee_point_definition='dummy cuff opening centre at the fixed CAD-derived distal mounting plane; pose target interpreted at this tool point',
        visual_mount_local_position=np.array([next(f['origin_m'] for f in meta['end_effector_frames'] if f['link']==w) for w in meta['wrists']]),
        head_reference_target=camera_p,source_start_frame=0,
        head_proxy_definition='front face surface at head mesh centre height; mounting proxy, not calibrated optical camera; no assumed headset thickness')
    del prime;torch.cuda.empty_cache()
    print('PRIME SAVED',body,clip,flush=True)
    return output

