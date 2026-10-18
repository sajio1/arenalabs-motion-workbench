"""Candidate fixed-window EgoAllo. Isolated from the production runner until seam audits pass."""
from pathlib import Path
import os, json, time, dataclasses
from egoallo_runtime import gpu_environment
os.environ.update(gpu_environment())
import jax
if jax.default_backend() != 'gpu':
    raise RuntimeError('CUDA JAX is required')
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.inference_utils import load_denoiser
from egoallo.hand_detection_structs import AriaHandWristPoseWrtWorld, CorrespondedAriaHandWristPoseDetections
from egoallo.transforms import SO3
from window_sampling import run_sampling_with_stitching, quadratic_ts
from window_plan import POLICY, plan, indices, digest, save_json

def run():
    root=Path(os.environ['ARENA_EGO_WORKDIR'])
    output=Path(os.environ.get('ARENA_WINDOW_OUTPUT',str(root/'windowed-v1')))
    output.mkdir(parents=True,exist_ok=True)
    source_hash=digest(root/'observations.npz')
    d=np.load(root/'observations.npz'); n=len(d['timestamps_s'])
    first_step=int(os.environ.get('ARENA_WINDOW_TEST_FIRST_STEP','240'))
    windows=list(plan(n,first_step=first_step))
    code_hash={name:digest(Path(__file__).with_name(name)) for name in ['run_egoallo_windowed.py','window_sampling.py','window_guidance.py','window_plan.py']}
    manifest=dict(policy=POLICY,source_sha256=source_hash,code_sha256=code_hash,windows=windows,
                  sampling_steps=30,inner_guidance=True,post_guidance=True,floor_z=float(d['floor_z']),
                  model_sha256=digest(root/'model.npz'))
    manifest_path=output/'manifest.json'
    normalized=json.loads(json.dumps(manifest))
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != normalized:
        raise RuntimeError('Checkpoint provenance changed; use a new output directory')
    save_json(manifest_path,manifest)
    torch.set_num_threads(8);device=torch.device('cuda:0')
    T=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    body=fncsmpl.SmplhModel.load(root/'model.npz').to(device)
    model=load_denoiser(root/'egoallo_checkpoint_april13/checkpoints_3000000').to(device)
    assert len(quadratic_ts())==31
    cpf=np.concatenate([Rotation.from_matrix(d['head_rotation']).as_quat()[:,[3,0,1,2]],d['head_position']],-1)
    assert len(cpf)==n+1
    forward=d['palm_forward'] if 'palm_forward' in d else d['palm_position']-d['wrist_position']
    normal=d['palm_normal']
    for key,value in [('CPF',cpf),('forward',forward),('normal',normal),('wrist',d['wrist_position'])]:
        if not np.isfinite(value).all(): raise ValueError(f'Non-finite {key}')
    if (np.linalg.norm(forward,axis=-1)<1e-6).any() or (np.linalg.norm(np.cross(forward,normal),axis=-1)<1e-6).any():
        raise ValueError('Missing/degenerate observed tool direction; no guessed orientation allowed')
    assembled_q=np.zeros((n,51,4),np.float32);assembled_q[...,0]=1
    assembled_contacts=np.zeros((n,21),np.float32)
    previous_end=0;fixed=None;seams=[]
    start_time=time.time()

    def sample(start,valid,context,fixed_shape,seed):
        idx=indices(start,valid);head_idx=np.r_[start,idx+1]
        hands=[]
        for side in range(2):
            hands.append(AriaHandWristPoseWrtWorld(confidence=T(context['valid']),
                wrist_position=T(d['wrist_position'][idx,side]),palm_position=T(d['palm_position'][idx,side]),
                wrist_normal=T(normal[idx,side]),palm_normal=T(normal[idx,side]),indices=torch.arange(300,device=device)))
        torch.manual_seed(seed)
        with torch.no_grad():
            traj=run_sampling_with_stitching(model,body,'aria_wrist_only',True,True,T(cpf[head_idx]),
                float(d['floor_z']),None,CorrespondedAriaHandWristPoseDetections(*hands),1,device,
                guidance_verbose=False,window_context=context,fixed_betas=None if fixed_shape is None else T(fixed_shape),frame_ids=idx)
            q=SO3.from_matrix(torch.cat([traj.body_rotmats,traj.hand_rotmats],dim=2)).wxyz.numpy(force=True)[0]
            arrays=dict(local_quats=q,betas=traj.betas.numpy(force=True)[0],contacts=traj.contacts.numpy(force=True)[0])
            if not all(np.isfinite(a).all() for a in arrays.values()):raise FloatingPointError('Non-finite final guidance')
            return arrays

    def context_for(start,valid,overlap):
        idx=indices(start,valid)
        ctx=dict(valid=(np.arange(300)<valid).astype(np.float32),forward=forward[idx].astype(np.float32),
                 anchor_mask=np.zeros(300,np.float32),anchor_quats=np.tile([1.,0,0,0],(300,21,1)).astype(np.float32),
                 anchor_positions=np.zeros((300,21,3),np.float32))
        if overlap:
            ctx['anchor_mask'][:overlap]=1
            ctx['anchor_quats'][:overlap]=assembled_q[start:start+overlap,:21]
            with torch.no_grad():
                posed=body.with_shape(T(fixed)).with_pose(T(np.tile([1.,0,0,0,0,0,0],(overlap,1))),T(assembled_q[start:start+overlap]))
                rp=fncsmpl_extensions.get_T_world_root_from_cpf_pose(posed,T(cpf[start+1:start+overlap+1]))
                ctx['anchor_positions'][:overlap]=posed.with_new_T_world_root(rp).Ts_world_joint.numpy(force=True)[:,:21,4:7]
        return ctx

    for wi,(start,valid) in enumerate(windows):
        overlap=max(0,previous_end-start);ctx=context_for(start,valid,overlap)
        if wi==0:
            calibration=output/'shape-calibration.npz'
            if not calibration.exists():
                arrays=sample(start,valid,ctx,None,20261013)
                np.savez_compressed(calibration,**arrays)
            with np.load(calibration) as cal: fixed=cal['betas'][:valid].mean(axis=0)
            save_json(output/'shape.json',dict(betas=fixed.tolist(),calibration_sha256=digest(calibration)))
        path=output/f'window-{wi:05d}.npz';receipt=path.with_suffix('.json')
        dependency=digest(output/f'window-{wi-1:05d}.npz') if wi else digest(output/'shape-calibration.npz')
        if path.exists() and receipt.exists():
            r=json.loads(receipt.read_text())
            if r['sha256']!=digest(path) or r['dependency_sha256']!=dependency:raise RuntimeError('Checkpoint chain changed')
            with np.load(path) as saved:arrays={k:saved[k] for k in saved.files}
        else:
            save_json(output/'progress.json',dict(stage='inference',window=wi+1,total_windows=len(windows),completed_frames=previous_end,total_frames=n))
            arrays=sample(start,valid,ctx,fixed,20261013+start)
            temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)
            save_json(receipt,dict(sha256=digest(path),dependency_sha256=dependency,start=start,valid_frames=valid,padding_frames=300-valid))
        q=arrays['local_quats'][:valid].copy();contacts=arrays['contacts'][:valid].copy()
        if overlap:
            old=assembled_q[start:previous_end].copy();new=q[:overlap]
            dot=(old*new).sum(-1,keepdims=True);new=np.where(dot<0,-new,new)
            # Geodesic crossfade after joint constrained inference; no source position editing.
            alpha=np.linspace(0,1,overlap,dtype=np.float32)[:,None,None]
            theta=np.arccos(np.clip((old*new).sum(-1,keepdims=True),-1,1))
            denom=np.maximum(np.sin(theta),1e-7)
            mixed=np.sin((1-alpha)*theta)/denom*old+np.sin(alpha*theta)/denom*new
            mixed=np.where(theta<1e-5,(1-alpha)*old+alpha*new,mixed)
            q[:overlap]=mixed/np.linalg.norm(mixed,axis=-1,keepdims=True)
            contacts[:overlap]=(1-alpha[:,:,0])*assembled_contacts[start:previous_end]+alpha[:,:,0]*contacts[:overlap]
            seams.append(dict(start=start,end=previous_end,overlap=overlap,raw_overlap_joint_angle_mean_deg=float(np.degrees(theta*2).mean())))
        assembled_q[start:start+valid]=q;assembled_contacts[start:start+valid]=contacts
        previous_end=start+valid
        save_json(output/'progress.json',dict(stage='window_saved',window=wi+1,total_windows=len(windows),completed_frames=previous_end,total_frames=n))
    # Fixed-size FK export keeps GPU allocation independent of total duration.
    chunks={k:[] for k in ['root_pose','joint_poses','vertices']}
    with torch.no_grad():
        for start in range(0,n,300):
            count=min(300,n-start);idx=indices(start,count)
            posed=body.with_shape(T(fixed)).with_pose(T(np.tile([1.,0,0,0,0,0,0],(300,1))),T(assembled_q[idx]))
            rp=fncsmpl_extensions.get_T_world_root_from_cpf_pose(posed,T(cpf[idx+1]));posed=posed.with_new_T_world_root(rp)
            for key,val in [('root_pose',rp),('joint_poses',posed.Ts_world_joint),('vertices',posed.lbs().verts)]:
                chunks[key].append(val.numpy(force=True)[:count])
    final={k:np.concatenate(v) for k,v in chunks.items()}
    np.savez_compressed(output/'egoallo_smplh.npz',**final,local_quats=assembled_q,contacts=assembled_contacts,
        betas=np.broadcast_to(fixed,(n,16)),faces=body.faces.numpy(force=True),cpf=cpf[1:],timestamps_s=d['timestamps_s'])
    assert digest(root/'observations.npz')==source_hash
    save_json(output/'seams.json',seams)
    save_json(output/'progress.json',dict(stage='complete',completed_frames=n,total_frames=n,elapsed_seconds=time.time()-start_time,
        output_sha256=digest(output/'egoallo_smplh.npz'),source_unchanged=True,validation_status='seam_and_robot_audits_pending'))

if __name__=='__main__':run()
