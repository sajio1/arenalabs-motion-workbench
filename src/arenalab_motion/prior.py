"""Immutable Prime outputs and exact native projection/export, shared by every source."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import time
import numpy as np
C = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
POLICY = 'full-body-direct-prime-native-projection-v1'

def configure_imports():
    # The installed package supplies all Python modules. Closed assets stay local.
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'

def sha(path):
    path = Path(path)
    return _sha(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)


@lru_cache(maxsize=128)
def _sha(path, size, modified):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def _arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


def _write_new(path, arrays):
    path = Path(path)
    if path.exists(): raise FileExistsError('Immutable output already exists: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: np.savez_compressed(f, **arrays)


def _receipt(path, value):
    with Path(path).open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-9): raise ValueError('Degenerate semantic frame')
    return v/n


def palm_frame(distal, normal):
    d = _unit(np.asarray(distal, float))
    n = _unit(np.asarray(normal, float)-d*np.sum(d*normal, axis=-1, keepdims=True))
    return np.stack([d, np.cross(n, d), n], axis=-1)


def _sync(codec):
    import torch
    if str(codec.device).startswith('cuda'): torch.cuda.synchronize()


def load_target(row):
    """Exact Grid native descriptor/angle-slot binding; CPU only."""
    configure_imports()
    import mujoco
    from alphamotion.embodiment.registry import load_from_xml
    from alphamotion.embodiment.mjcf_build import build_merge
    row=row.get('pipeline_row',row)
    xml=str(row.get('model') or row['xml'])
    m=mujoco.MjModel.from_xml_path(xml);emb=load_from_xml(xml,row['body'])
    keep,frame=build_merge(m)
    free=np.flatnonzero(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    if len(free)!=1 or m.jnt_qposadr[free[0]]!=0: raise ValueError('Adapter requires one native floating root at qpos 0')
    tab=np.full((emb.spec.J,3),-1,int)
    for j,names in enumerate(emb.qnames):
        for k,name in enumerate(names):
            ji=m.joint(name).id
            if m.jnt_type[ji]!=mujoco.mjtJoint.mjJNT_HINGE: raise ValueError('Only scalar hinge angle slots are supported')
            tab[j,k]=m.jnt_qposadr[ji]
    scalar=set(m.jnt_qposadr[m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE].tolist())
    if scalar!=set(tab[tab>=0].tolist()): raise ValueError('Native descriptor does not cover all hinge qpos addresses')
    return dict(row=row,model=m,emb=emb,spec=emb.spec,dof=emb.dof,rest=emb.rest,
                keep=np.asarray(keep),frame=frame,prior_body_ids=np.asarray([frame[b] for b in keep]),
                qpos_slot_map=tab,xml=xml,model_sha256=sha(xml))


def task_goals(row, source):
    row=row.get('pipeline_row',row)
    local=np.stack([palm_frame(np.asarray(row['palm_frames'][s+'_wrist']['distal']),
                              np.asarray(row['palm_frames'][s+'_wrist']['normal'])) for s in ['left','right']])
    desired=source['palm_R_native']@local.transpose(0,2,1)[None]
    return dict(goal_body_R_native=desired,goal_wrist_position_native=source['obs_wrist_zup_m'].copy(),
                semantic_palm_local_native=local,operation_local_native_m=np.zeros((source['frames'],2,3)),
                operation_point_semantics='Original no-hand source wrist/link origin; dynamic tool offsets must be supplied explicitly')


def native_to_motion(row, qpos_world, outputdir=None, goal_R_native=None, *, source=None,
                     operation_local_native_m=None):
    """Exact native FK export; no root alignment, smoothing or pose modification."""
    configure_imports()
    import mujoco
    from scipy.spatial.transform import Rotation
    target=load_target(row);m=target['model'];data=mujoco.MjData(m);qpos=np.asarray(qpos_world,dtype=np.float64)
    if qpos.ndim!=2 or qpos.shape[1]!=m.nq or not np.isfinite(qpos).all(): raise ValueError('Invalid native qpos')
    T=len(qpos);q=np.zeros((T,target['spec'].J,3));tab=target['qpos_slot_map'];a,b=np.where(tab>=0)
    q[:,a,b]=qpos[:,tab[a,b]]
    row=target['row']
    wrists=np.array([m.body(row['endpoints'][side+'_wrist']).id for side in ('left','right')])
    wrist_indices=[]
    for bid in wrists:
        matches=np.flatnonzero(target['prior_body_ids']==bid)
        if not len(matches):matches=np.flatnonzero(target['keep']==bid)
        if len(matches)!=1:raise ValueError('Wrist endpoint has no unique native descriptor index')
        wrist_indices.append(int(matches[0]))
    P=[];R=[];root=[];wristP=[];wristR=[]
    for pose in qpos:
        data.qpos[:]=pose;mujoco.mj_forward(m,data)
        P.append(data.xpos[target['keep']].copy());R.append(data.xmat[target['prior_body_ids']].reshape(-1,3,3).copy())
        root.append(data.xpos[target['keep'][0]].copy())
        wristP.append(data.xpos[wrists].copy());wristR.append(data.xmat[wrists].reshape(2,3,3).copy())
    P=np.asarray(P);R=np.asarray(R);Ry=C@R@C.T;root=np.asarray(root)
    wristP=np.asarray(wristP);wristR=np.asarray(wristR)
    floor=float(source['floor']) if source is not None else 0.
    from .geometry import UniversalOracle
    minimum,feet=UniversalOracle(row).surface.exact_heights(qpos)
    rot6d=Ry[...,:,:2].swapaxes(-1,-2).reshape(T,-1,6)
    arrays=dict(qpos=qpos,qpos_world=qpos,q=q,rot6d=rot6d,joints_yup_cm=P@C.T*100,
                world_position_cm=P@C.T*100,constraint_joint_indices=np.asarray(wrist_indices),
                sole_surface_height_cm=(feet-floor)*100,all_surface_height_cm=(minimum-floor)*100,
                floor_z_m=np.array(floor),ground_height_m=np.array(floor),
                wrist_body_position_native_m=wristP,wrist_body_rotation_native=wristR,
                root_t_yup_cm=root@C.T*100,root_t=root@C.T*100,
                native_body_position_m=P,native_body_rotation=R,prior_body_ids=target['prior_body_ids'],
                descriptor_body_ids=target['keep'],qpos_slot_map=tab,
                root_quaternion_wxyz=Rotation.from_matrix(R[:,0]).as_quat(scalar_first=True),
                joint_names=np.asarray(target['spec'].joint_names),parents=target['spec'].parents,
                fps=np.array(30.))
    if source is not None:
        arrays.update(timestamps_s=source['times'],floor_z_m=np.array(source['floor']),
                      goal_wrist_position_native=source['obs_wrist_zup_m'],
                      observed_wrist_position_yup=source['obs_wrist_zup_m']@C.T,
                      source_timestamps_s=source['times'],source_role_indices=source['roles'],
                      source_root_cm=source['position_yup_m'][:,0]*100)
    if goal_R_native is not None:
        goal_R_native=np.asarray(goal_R_native)
        if goal_R_native.shape!=(T,2,3,3):raise ValueError('Expected T x 2 wrist rotation goals')
        error_R=wristR.swapaxes(-1,-2)@goal_R_native
        arrays.update(goal_body_R_native=goal_R_native,observed_wrist_rotation_yup=C@goal_R_native@C.T,
                      wrist_error_deg=np.rad2deg(np.arccos(np.clip((np.trace(error_R,axis1=-2,axis2=-1)-1)/2,-1,1))))
    if operation_local_native_m is not None:
        local=np.asarray(operation_local_native_m)
        if local.shape!=(T,2,3):raise ValueError('Expected T x 2 operation points in native wrist-link local coordinates')
        arrays.update(operation_local_native_m=local,operation_local_yup_m=local@C.T,
                      operation_world_position_native_m=wristP+np.einsum('tsij,tsj->tsi',wristR,local))
    if outputdir is not None:_write_new(Path(outputdir)/'motion.npz',arrays)
    return arrays


def infer_prior(row, source, outputdir, codec=None, chunk=32, projection_iterations=12):
    """Return immutable neural prior and projected *initial* native qpos, not a refined solution."""
    configure_imports()
    import torch
    from scipy.spatial.transform import Rotation
    from alphamotion.engine.nets.rotations import rot6d_to_matrix
    from greenwich_motion_sdk.native_projection import project
    row=row.get('pipeline_row',row);folder=Path(outputdir);folder.mkdir(parents=True,exist_ok=True)

    if codec is None:raise ValueError('An explicitly configured local Prime codec is required')
    if 'codes' not in source:encode_source(source,folder/'source_prior_fullbody',codec,chunk)
    target=load_target(row);goals=task_goals(row,source)
    rawpath=folder/'prime_original.npz';nativepath=folder/'native_prior.npz';receiptpath=folder/'prior_receipt.json'
    ident=dict(policy=POLICY,source_hashes=source['source_hashes'],model_sha256=codec._motion_model_identity,
               native_xml_sha256=target['model_sha256'],projection_iterations=projection_iterations)
    if rawpath.exists():
        receipt=json.loads(receiptpath.read_text())
        if receipt['identity']!=ident or sha(rawpath)!=receipt['raw_sha256'] or sha(nativepath)!=receipt['native_sha256']:
            raise ValueError('Immutable target prior identity changed; use a new output directory')
        arrays=_arrays(nativepath)
    else:
        spec=target['spec'];codes=torch.as_tensor(source['codes'],device=codec.device);T=source['frames']
        _sync(codec);start=time.perf_counter()
        with torch.inference_mode():
            outputs=[codec.decode_full(codes[i:i+chunk],spec,target['dof'],contact=True) for i in range(0,T,chunk)]
        _sync(codec);decode_s=time.perf_counter()-start
        raw,pos,contact=[torch.cat([x[k] for x in outputs]) for k in range(3)]
        rawarrays=dict(codes=source['codes'],raw_rot6d=raw.cpu().numpy(),decoder_position=pos.cpu().numpy(),
                       model_contact=contact.cpu().numpy(),joint_names=np.asarray(spec.joint_names),fps=np.array(30.))
        _write_new(rawpath,rawarrays)
        start=time.perf_counter()
        q,_,projection=project(raw.double(),spec,torch.as_tensor(target['dof'],device=codec.device,dtype=torch.float64),
                               torch.as_tensor(target['rest'],device=codec.device,dtype=torch.float64),iterations=projection_iterations)
        _sync(codec);projection_s=time.perf_counter()-start
        start=time.perf_counter()
        q=q.detach().cpu().numpy();Rroot=rot6d_to_matrix(raw[:,0].double()).cpu().numpy()
        native=np.broadcast_to(target['model'].qpos0,(T,target['model'].nq)).copy()
        a,b=np.where(target['qpos_slot_map']>=0);native[:,target['qpos_slot_map'][a,b]]=q[:,a,b]
        root=source['position_yup_m'][:,0].copy();root[:,1]=source['floor']+pos[:,0,0].cpu().numpy()
        native[:,:3]=root@C;native[:,3:7]=Rotation.from_matrix(C.T@Rroot@C).as_quat(scalar_first=True)
        arrays=native_to_motion(row,native,goal_R_native=goals['goal_body_R_native'],source=source,
                                operation_local_native_m=goals['operation_local_native_m'])
        arrays.update(raw_rot6d=rawarrays['raw_rot6d'],decoder_position=rawarrays['decoder_position'])
        native_conversion_s=time.perf_counter()-start
        _write_new(nativepath,arrays)
        receipt=dict(identity=ident,raw_sha256=sha(rawpath),native_sha256=sha(nativepath),
                     decode_s=decode_s,native_projection_s=projection_s,projection=projection,
                     native_conversion_cpu_and_copy_s=native_conversion_s,
                     cold_model_load_s=float(getattr(codec,'_grid_cold_load_s',0.)),
                     phase_backends=dict(decode='GPU neural network; cold descriptor/semantic preparation can include CPU',native_projection='GPU Torch batched LM, CPU orchestration',native_conversion='D2H plus CPU exact MuJoCo FK and named joint packing'),
                     root_policy='Measured source pelvis horizontal position; Prime decoder height above unchanged source floor; decoded root orientation',
                     native_fk='Exact MuJoCo model and mapped scalar joint addresses; no display fit',
                     source_completion_sha256=source['inferred_prior_receipt']['derived_sha256'],
                     physical_validation=False,refined=False)
        _receipt(receiptpath,receipt)
    return dict(**arrays,**{k:v for k,v in goals.items() if k not in arrays},target=target,
                receipt=receipt,raw_path=str(rawpath),native_path=str(nativepath))



def encode_source(source,outputdir,codec,chunk=32):
    import torch
    from alphamotion.engine.nets.rotations import matrix_to_rot6d
    folder=Path(outputdir);folder.mkdir(parents=True,exist_ok=True)
    raw_path=folder/'source_prime_original.npz'
    body_path=folder/'source_body_original.npz'
    receipt_path=folder/'source_prior.json'
    identity=dict(policy=POLICY,source_hashes=source['source_hashes'],model_sha256=codec._motion_model_identity)
    if raw_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['identity']!=identity or sha(raw_path)!=receipt['raw_sha256'] or sha(body_path)!=receipt['derived_sha256']:
            raise ValueError('Immutable supplied-body encoding changed; use a new run')
        raw=_arrays(raw_path)
    else:
        p=source['full_body_position_yup_m']*100
        relative=p-p[:,:1]
        reach=float(np.linalg.norm(relative,axis=-1).max())
        if reach<1e-6:raise ValueError('Empty supplied body reach')
        pose=np.concatenate([matrix_to_rot6d(torch.as_tensor(source['full_body_rotation_yup'],dtype=torch.float32)).numpy(),
                             (relative/reach).astype(np.float32)],-1)
        _sync(codec);start=time.perf_counter()
        with torch.inference_mode():
            codes=torch.cat([codec.encode(torch.as_tensor(pose[i:i+chunk],device=codec.device),source['spec'],source['source_dof'])
                             for i in range(0,len(pose),chunk)])
        _sync(codec);elapsed=time.perf_counter()-start
        raw=dict(codes=codes.cpu().numpy(),input_pose9=pose,input_reach_cm=np.array(reach),
                 observed_nodes=np.arange(len(source['names'])),encoder_policy=np.array(POLICY),source_decoder_used=np.array(False),
                 source_frame_indices=source['source_frame_indices'],timestamps_s=source['times'])
        _write_new(raw_path,raw)
        _write_new(body_path,dict(position_world_m=source['inferred_world_pos_zup'],
            rotation_world=source['inferred_world_R_zup'],bone_directions=source['inferred_bone_directions'],
            source_frame_indices=source['source_frame_indices'],timestamps_s=source['times']))
        receipt=dict(identity=identity,raw_sha256=sha(raw_path),derived_sha256=sha(body_path),
                     encoder_policy=POLICY,encode_s=elapsed,source_decode_s=0.,source_decoder_used=False,
                     input_reach_cm=reach,observed_joint_count=len(source['names']),source_body_modified=source['source_body_modified'],
                     source_body_modification=('Uniform similarity from local robot limb dimensions; rotations and timestamps unchanged' if source['source_body_modified'] else 'None: supplied complete body encoded directly'),
                     morphology_scale=source['morphology_scale'],
                     human_model=source['source_body_provenance'],cold_model_load_s=float(getattr(codec,'_grid_cold_load_s',0.)),
                     phase_timing_semantics='Synchronized phase wall time, including descriptor work when cold')
        _receipt(receipt_path,receipt)
    source.update(codes=raw['codes'],input_pose9=raw['input_pose9'],inferred_prior_receipt=receipt,source_prior_dir=str(folder))
