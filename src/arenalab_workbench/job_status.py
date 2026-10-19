"""Queue status snapshots that tolerate a worker starting during a read."""
import json,time

def job_view(job,jobs,folder):
    path=folder/job['id']/'state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(status='queued',phase='Waiting for GPU')
    result={**job,**state}
    if result['status']=='running' and result.get('started_at'):
        result['elapsed_seconds']=time.time()-result['started_at']
    if result['status']=='queued':
        # Do not re-read this job to find its index: it can already be running.
        # The snapshot may be queued for one poll, but must still be valid.
        position=1
        for previous in jobs.values():
            if previous['id']==job['id']:break
            if json.loads((folder/previous['id']/'state.json').read_text()).get('status')=='queued':position+=1
        result['queue_position']=position
    result['stages']={p.stem:json.loads(p.read_text()) for p in path.parent.glob('*.json') if p.name not in {'state.json','job.json'}}
    if result['status'] in {'queued','running'}:result['elapsed_seconds']=time.time()-job['created_at']
    from arenalab_workbench.gpu_scheduler import ROOT
    gpu_stages=[];all_waiting=[]
    for directory in ['waiting','events']:
        for p in (ROOT/directory).glob('*.json'):
            try:value=json.loads(p.read_text())
            except (FileNotFoundError,json.JSONDecodeError):continue
            if directory=='waiting':all_waiting.append(value)
            if value.get('job')==job['id']:gpu_stages.append(value)
    result['gpu_stages']=gpu_stages
    active=[v for v in gpu_stages if v['state']=='running']
    waiting=[v for v in gpu_stages if v['state']=='queued']
    labels={'retarget_robot.py':'Robot refinement','gripper_retarget.py':'Body and gripper retargeting','segment_scene.py':'Object segmentation','scene_inference.py':'Scene perception','gripper_render.py':'Robot video','render_recording.py':'Robot video','contact_recording.py':'Finger contact','gripper_export.py':'Robot export'}
    if active and result['status']=='running':
        result['phase']=' · '.join(f"GPU {v['gpu']}: {labels.get(v['label'],v['label'])}" for v in active)
    if waiting and not active and result['status']=='running':
        position=next(i+1 for i,v in enumerate(sorted(all_waiting,key=lambda v:v['queued_at'])) if v.get('job')==job['id'])
        result.update(status='queued',phase='Waiting for an available GPU',queue_position=position)
    return result
