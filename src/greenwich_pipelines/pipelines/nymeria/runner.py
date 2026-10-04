"""Actual V3 production branch; earlier V22 experiments are not mixed in."""
import numpy as np
from greenwich_pipelines.runtime import load_model
from greenwich_pipelines.artifacts import artifact,sha256,write_json,inspect_native
from .input import read_source,windows
from .completion import complete
from .embodiment_exact import load_robot
from . import posture

def run(job,robots):
    import torch
    torch.set_num_threads(4)
    source=read_source(job.source);prime=load_model();results=[];skipped=[]
    for start,end,segment in windows(source,job.window_frames):
        if segment is None:skipped.append(dict(start=start,end=end,reason='tail_under_3_frames'));continue
        label=f'{job.id}__f{start:07d}'
        saved_source=job.output/'segments'/label/'source.npz';saved_source.parent.mkdir(parents=True)
        np.savez_compressed(saved_source,**{k:v for k,v in segment.items() if k!='clock_report'})
        results.append(artifact(job.output,saved_source,'source',clip=label,frame_start=start,frame_end=end,coordinate_frame='right-handed-y-up',position_unit='cm'))
        for robot in robots:
            if 'palm_frames' not in robot:raise ValueError('Nymeria requires fixed palm_frames physical metadata')
            body=load_robot(robot);folder=saved_source.parent/robot['body'];folder.mkdir()
            reference=complete(prime,body,robot,segment)
            raw=folder/'prime_original.npz';np.savez_compressed(raw,**reference);digest=sha256(raw)
            flags=np.asarray(segment['foot_contacts'],dtype=bool)
            contacts=np.stack((flags[:,[0,0,1,1]],flags[:,[2,2,3,3]]),1)
            result,detail=posture.fit(prime,body,reference,reference,contacts,30.)
            if sha256(raw)!=digest:raise RuntimeError('Original model prior changed')
            result.update(raw_model_reference=str(raw),raw_model_reference_sha256=digest,seed_is_model_prior=False,fps=np.array(30.))
            path=folder/'result.npz';np.savez_compressed(path,**result);inspect_native(path,'nymeria')
            ids=np.asarray(result['endpoint_indices']);error=np.linalg.norm(result['joints_yup_cm'][:,ids]-result['input_positions_yup_cm'],axis=-1)*10
            report=dict(policy=posture.POLICY,fps=30,source_frame_range=[start,end],clock=segment['clock_report'],wrist_position_max_mm=error[:,:2].max(0).tolist(),quality_status='not_evaluated',**detail)
            write_json(folder/'report.json',report)
            results.extend([artifact(job.output,raw,'model_prior',clip=label,robot=robot['body']),artifact(job.output,path,'motion',clip=label,robot=robot['body'],frames=end-start,coordinate_frame='right-handed-y-up',position_unit='cm',joint_encoding='descriptor-angle-slots'),artifact(job.output,folder/'report.json','report')])
    write_json(job.output/'source-clock.json',dict(clock=source['clock_report'],skipped_ranges=skipped,window_policy='independent windows; no seam-continuity claim'))
    if not any(r['kind']=='motion' for r in results):raise RuntimeError('No valid source windows')
    return results
