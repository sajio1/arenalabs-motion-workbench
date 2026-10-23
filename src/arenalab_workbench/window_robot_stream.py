"""Consume committed human windows on a separate GPU, ahead of final refinement."""
from arenalab_workbench.settings import paths
from pathlib import Path
import os,sys,json,time,subprocess,shutil
import numpy as np
from arenalab_workbench.window_plan import digest,save_json


def committed(output,total):
    manifest=json.loads((output/'manifest.json').read_text());q=np.zeros((total,51,4),np.float32);q[...,0]=1
    contacts=np.zeros((total,21),np.float32);end=0;stable=0
    for wi,(start,valid) in enumerate(manifest['windows']):
        p=output/f'window-{wi:05d}.npz';receipt=p.with_suffix('.json')
        if not receipt.exists():break
        assert json.loads(receipt.read_text())['sha256']==digest(p)
        with np.load(p) as d:new=d['local_quats'][:valid].copy();con=d['contacts'][:valid].copy()
        overlap=max(0,end-start)
        if overlap:
            old=q[start:end].copy();current=new[:overlap];dot=(old*current).sum(-1,keepdims=True);current=np.where(dot<0,-current,current)
            alpha=np.linspace(0,1,overlap,dtype=np.float32)[:,None,None];theta=np.arccos(np.clip((old*current).sum(-1,keepdims=True),-1,1));denom=np.maximum(np.sin(theta),1e-7)
            mixed=np.sin((1-alpha)*theta)/denom*old+np.sin(alpha*theta)/denom*current;mixed=np.where(theta<1e-5,(1-alpha)*old+alpha*current,mixed)
            new[:overlap]=mixed/np.linalg.norm(mixed,axis=-1,keepdims=True)
            con[:overlap]=(1-alpha[:,:,0])*contacts[start:end]+alpha[:,:,0]*con[:overlap]
        q[start:start+valid]=new;contacts[start:start+valid]=con;end=start+valid
        stable=end if end==total else manifest['windows'][wi+1][0]
    return q,contacts,stable


def worker(root,rid):
    import torch
    from scipy.spatial.transform import Rotation
    from greenwich_pipelines.runtime import configure,load_model
    from greenwich_motion_sdk.motion import MotionClip
    from alphamotion.engine.nets.rotations import SkeletonSpec
    from alphamotion.embodiment.dof_features import ball_dof
    from alphamotion.engine.descriptor import build_from_mjcf
    from alphamotion.engine import constraints as c
    from greenwich_pipelines.input_policy import encode_observed,POLICY
    from greenwich_pipelines.frame_cache import target_key
    root=Path(root);output=root/'windowed-v1';dest=root/'frame-stream'/rid;dest.mkdir(parents=True,exist_ok=True)
    started=time.time();save_json(dest/'progress.json',dict(stage='started',started_at=started,committed_frames=0))
    configure(str(paths.config / 'runtime.cloud.json'));torch.set_num_threads(2);device=torch.device('cuda:0')
    codec=load_model()
    rows={r['id']:r for r in json.loads(paths.viewer_config.read_text())['robots']}
    registry=paths.store / 'uploaded/registry.json'
    if registry.exists():rows.update({r['id']:r for r in json.loads(registry.read_text())})
    row=rows[rid]['pipeline_row'];spec,dof,*_=build_from_mjcf(row['xml'],row['body']);key=target_key(codec,spec,dof)
    obs=np.load(root/'observations.npz');total=len(obs['timestamps_s']);cpf=np.concatenate([Rotation.from_matrix(obs['head_rotation']).as_quat()[:,[3,0,1,2]],obs['head_position']],-1)
    assert len(cpf)==total+1
    T=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    done=0;events=[]
    while done<total:
        if not (output/'shape.json').exists() or not (output/'manifest.json').exists():time.sleep(.3);continue
        q,contacts,stable=committed(output,total)
        if stable<=done:time.sleep(.3);continue
        start=done;end=min(stable,start+300);folder=dest/f'chunk-{start:06d}-{end:06d}';folder.mkdir(exist_ok=True);begin=time.time()
        if not (folder/'robot.npz').exists():
            beta=np.asarray(json.loads((output/'shape.json').read_text())['betas'],np.float32);count=end-start
            np.savez_compressed(folder/'committed-input.npz',local_quats=q[start:end],betas=beta)
            for name in ['model.npz']:
                if not (folder/name).exists():(folder/name).symlink_to(root/name)
            original=json.loads((root/'input_manifest.json').read_text());save_json(folder/'input_manifest.json',{**original,'frames':count})
            env=os.environ.copy();env['ARENA_EGO_WORKDIR']=str(folder)
            with (folder/'native-h-export.log').open('w') as log:subprocess.run([str(paths.egoallo_python),str(Path(__file__).with_name('emit_body_window.py')),str(root),str(folder),str(start),str(end)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            clip=MotionClip.load(folder/'canonical.npz');human=SkeletonSpec('stream_'+str(start),clip.parents,clip.rest_offsets_cm.astype(np.float32),clip.names);hd,_=ball_dof(len(clip.names));gp=clip.world_position_cm-clip.world_position_cm[:,:1]
            r6=c.matrix_to_rot6d(torch.tensor(clip.global_rotation,dtype=torch.float32));pose=torch.cat([r6,torch.tensor(gp/np.linalg.norm(gp,axis=-1).max(),dtype=torch.float32)],-1).to(device)
            roles=[clip.names.index(k) for k in ['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']]
            with torch.no_grad():
                codes=torch.cat([encode_observed(codec,pose[k:k+32],human,hd,roles) for k in range(0,count,32)])
                out=codec.decode_full(codes,spec,dof,contact=True)
            temp=folder/'robot.tmp.npz';np.savez_compressed(temp,target_key=np.array(key),encoder_policy=np.array(POLICY),frame_ids=np.arange(start,end),timestamps_s=obs['timestamps_s'][start:end],codes=codes.cpu().numpy(),raw_rot6d=out[0].cpu().numpy(),decoder_position=out[1].cpu().numpy(),model_contact=out[2].cpu().numpy());temp.replace(folder/'robot.npz')
        done=end;body_progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else {}
        events.append(dict(start_frame=start,end_frame=end,started_at=begin,finished_at=time.time(),body_progress=body_progress))
        save_json(dest/'progress.json',dict(stage='complete' if done==total else 'streaming',started_at=started,committed_frames=done,total_frames=total,events=events,final_refinement='Pending original complete refinement; this is an initial model output'))
        if time.time()-started>3600:raise TimeoutError('Body window producer stalled')


if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
    else:
        from arenalab_workbench.gpu_scheduler import run
        env=os.environ.copy()
        for key in ['ARENA_GPU_LEASE','ARENA_GPU_LOCK_FD','CUDA_VISIBLE_DEVICES','EGL_DEVICE_ID']:env.pop(key,None)
        env['ARENA_GPU_POOL']='0,1,2,3'
        run([str(paths.motion_python),str(Path(__file__)), 'worker',sys.argv[1],sys.argv[2]],env=env,check=True)
