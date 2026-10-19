"""Single leased GPU: prebuild immutable human priors, never robot caches."""
from arenalab_workbench.settings import paths
from pathlib import Path
import json,os,subprocess,time,argparse,hashlib
from arenalab_workbench.gpu_scheduler import lease,resource_lock,write
from arenalab_workbench.scene_job import ROOT

def run(gpu):
    code=Path(__file__).parent;out=ROOT/'smpl-prebuild';out.mkdir(exist_ok=True)
    status=out/'progress.json';catalog=json.loads(paths.store / 'manipulation4d/catalog.json'.read_text())
    from arenalab_workbench.gripper_registry import sources
    from arenalab_workbench.height_registry import default_id,variant
    from arenalab_workbench.body_prior import folder
    from arenalab_workbench.gripper_body import body_path
    registry=sources();items=[]
    for entry in catalog:
        sid=entry['id'];effective=sid
        if entry['dataset']=='UMI':
            effective=default_id(sid)
            if not effective:
                items.append(dict(id=sid,dataset=entry['dataset'],status='needs_height'));continue
        if effective in registry:
            work=Path(registry[effective]['work']);dest=body_path(work);command=[str(paths.motion_python),str(code/'gripper_body.py'),str(work)]
        else:
            try:dest=folder(effective)
            except Exception as error:
                items.append(dict(id=sid,status='unavailable',error=str(error)));continue
            command=[str(paths.motion_python),str(code/'body_prior.py'),effective]
        items.append(dict(id=sid,effective_id=effective,dataset=entry['dataset'],work=str(dest),command=command,status='pending'))
    def report(state):
        write(status,dict(status=state,kind='Prebuilt SMPL',gpu=gpu,updated_at=time.time(),items=items,ready=sum(x['status']=='ready' for x in items),total=len(items),robot_cache=False))
    report('queued')
    env=os.environ.copy();env.update(ARENA_GPU_POOL=str(gpu),PYTHONPATH=paths.pythonpath(),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    with resource_lock('smpl-prebuild-worker'):
        with lease('Prebuilt SMPL batch',env) as selected:
            for item in items:
                if item['status']!='pending':continue
                sid=item['effective_id'];item.update(status='running',started_at=time.time());report('running')
                try:
                    with resource_lock('source:'+sid):
                        with (out/(hashlib.sha256(sid.encode()).hexdigest()[:16]+'.log')).open('a') as log:
                            # Pass the lease FD to keep reservation alive through descendants.
                            result=subprocess.run(item['command'],env=selected,stdout=log,stderr=subprocess.STDOUT,pass_fds=(int(selected['ARENA_GPU_LOCK_FD']),))
                    if result.returncode:raise RuntimeError('Body worker exited '+str(result.returncode))
                    work=Path(item['work']);ready=json.loads((work/'prior-ready.json').read_text())
                    for name,digest in ready['raw_sha256'].items():
                        if hashlib.sha256((work/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Prior checksum mismatch: '+name)
                    targets=work/'operation-accuracy/source-targets.npz'
                    if not targets.exists():raise RuntimeError('Missing immutable operation targets')
                    write(work/'prebuilt-smpl.json',dict(kind='Prebuilt SMPL',robot_cache=False,source_id=sid,prior=ready,operation_targets_sha256=hashlib.sha256(targets.read_bytes()).hexdigest(),prepared_at=time.time()))
                    item.update(status='ready',seconds=time.time()-item['started_at'])
                except Exception as error:item.update(status='failed',error=str(error),seconds=time.time()-item['started_at'])
                report('running')
    report('finished')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=3);run(p.parse_args().gpu)
