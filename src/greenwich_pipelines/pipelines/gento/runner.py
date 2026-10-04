"""Gento production recipe, arbitrary roster IDs and explicit model metadata."""
import json
from pathlib import Path
import numpy as np
from greenwich_pipelines.config import read_json,resolve
from greenwich_pipelines.artifacts import artifact,sha256,write_json,inspect_native

def prepare_metadata(row,output):
    path=Path(row['metadata']);metadata=read_json(path)
    for key in ('model','source'):metadata[key]=str(resolve(path.parent,metadata[key]))
    target=output/'models'/row['body']/'metadata.json';write_json(target,metadata)
    return metadata

def validate_fk(model,result,raw):
    import mujoco
    data=mujoco.MjData(model);positions=[];rotations=[]
    for q in result['qpos']:
        data.qpos[:]=q;mujoco.mj_forward(model,data)
        positions.append(data.xpos[raw['wrist_ids']].copy())
        rotations.append(data.xmat[raw['wrist_ids']].reshape(-1,3,3).copy())
    actual=np.asarray(positions)+np.einsum('nkij,kj->nki',np.asarray(rotations),raw['ee_local_position'])
    if not np.allclose(actual,result['actual_end_effector_positions'],atol=1e-7,rtol=0):raise ValueError('Independent saved-qpos TCP FK mismatch')
    if np.ptp(result['qpos'][:,:7],axis=0).max()!=0:raise ValueError('Fixed chassis moved')
    return dict(saved_qpos_tcp_max_disagreement_m=float(np.max(np.abs(actual-result['actual_end_effector_positions']))),base_max_change=0.)

def run(job,robots):
    import torch,mujoco
    from . import source as g,preparation as batch,refiner as ref
    torch.set_num_threads(1);torch.manual_seed(0)
    g.ROOT=batch.ROOT=ref.ROOT=job.output
    results=[]
    for row in robots:
        if 'metadata' not in row:raise ValueError('Gento requires a metadata file with mixed-joint and TCP definitions')
        metadata=prepare_metadata(row,job.output)
        body=row['body'];clip=job.id
        batch.RAW_VERSION='raw'
        rawpath=batch.prepare(body,clip,target_forward_m=.10,source_path=job.source,frames=None,require_video=False)
        with np.load(rawpath,allow_pickle=False) as d:raw={k:d[k].copy() for k in d.files}
        digest=sha256(rawpath);model=mujoco.MjModel.from_xml_path(metadata['model'])
        seed=raw['seed'].copy();seed[:,7:]=np.clip(seed[:,7:],model.jnt_range[1:,0],model.jnt_range[1:,1])
        for j in range(1,model.njnt):
            if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:seed[:,model.jnt_qposadr[j]]=model.jnt_range[j].mean()
        init=rawpath.parent/'numerical_seed.npz';np.savez_compressed(init,qpos=seed)
        ref.solve(body,clip,version='native',raw_path=rawpath,frames=len(raw['position']),head_position_reference=True,head_placement=True,global_height=True,solve_setback=False,head_motion_reference=True,initialization_path=init,shoulder_placement=True,scene_backoff_m=.05)
        folder=job.output/'native'/body/clip;path=folder/'result.npz'
        with np.load(path,allow_pickle=False) as d:result={k:d[k].copy() for k in d.files}
        if sha256(rawpath)!=digest:raise RuntimeError('Original model prior changed')
        if not np.array_equal(result['targets'],raw['position']) or not np.array_equal(result['target_orientation'],raw['orientation']):raise ValueError('Source task goals changed')
        result['ee_local_position']=raw['ee_local_position'];np.savez_compressed(path,**result)
        frames=inspect_native(path,'gento');report=read_json(folder/'report.json')
        report.update(independent_fk=validate_fk(model,result,raw),accepted=False,quality_status='not_evaluated',prior_sha256=digest,fps=30,task_translation_world_m=raw['task_translation_world_m'].tolist())
        write_json(folder/'report.json',report)
        results.extend([artifact(job.output,rawpath,'model_prior',robot=body,clip=clip),artifact(job.output,path,'motion',frames=frames,robot=body,clip=clip,coordinate_frame='right-handed-z-up',position_unit='m',joint_encoding='mujoco-qpos-wxyz-free-root'),artifact(job.output,folder/'report.json','report')])
    return results
