"""One-card, resumable human-motion asset build. No scene or robot generation.

Assets reference immutable existing body outputs, not robot-result caches. Run
against a frozen copy of the current body pipeline; leave the web service alone.
"""
from arenalab_workbench.settings import paths
from arenalab_workbench.human_reference import motion_path,family_for_work
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time, uuid

ROOT = paths.workbench
SCHEMA = 'arenalab.human-motion-asset.v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def identity(context):
    """Source, time window, calibration and inference define the human asset.

    There is deliberately no robot, hand model, DA3, SAM or video-render key.
    Height/floor calibration is an inference input and MUST invalidate identity.
    """
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def verified(work):
    work = Path(work)
    receipt = json.loads((work / 'prior-ready.json').read_text())
    required = {'canonical.npz', motion_path(work).name, 'egoallo_smplh.npz', 'observations.npz'}
    hashes = receipt.get('raw_sha256', {})
    if not required.issubset(hashes):
        raise ValueError('Body receipt is missing required immutable artifacts')
    for name, expected in hashes.items():
        if digest(work / name) != expected:
            raise ValueError('Immutable body artifact changed: ' + name)
    return receipt


def plan():
    import numpy as np
    from arenalab_workbench.body_prior import folder, POLICY, policy_for
    from arenalab_workbench.gripper_body import body_path
    from arenalab_workbench.gripper_registry import sources
    from arenalab_workbench.scene_job import resolve_clip
    catalog = json.loads((paths.store / 'manipulation4d/catalog.json').read_text())
    grippers = sources(); groups = {}; excluded = []; failed = []
    for entry in catalog:
        sid = entry['id']
        if entry['dataset'] == 'UMI':
            excluded.append(dict(id=sid, reason='UMI height is unspecified; excluded by user'))
            continue
        try:
            if entry['dataset'] == 'Real robot':
                source_work = Path(grippers[sid]['work'])
                meta = json.loads((source_work / 'source.json').read_text())
                with np.load(source_work / 'source.npz') as data:
                    frames = len(data['tcp_zup']); floor = float(data['floor_z'])
                work = body_path(source_work)
                context = dict(dataset=entry['dataset'], source_sha256=digest(source_work/'source.npz'),
                    frames=frames, fps=30, source_window=meta.get('source_frame_range'),
                    floor_z_m=floor, height_calibration=meta.get('height_calibration'),
                    head_calibration=meta.get('head_calibration'), policy='gripper-head-two-ee-smplh-v1')
                kind = 'gripper'
            else:
                public, row = resolve_clip(sid); work = folder(sid); source_work = None
                context = dict(dataset=entry['dataset'], source_sha256=digest(row['source']),
                    frames=int(row['frames']), fps=row['fps'], source_window=row.get('source_window'),
                    floor_z_m=float(public.get('height_calibration', {}).get('floor_z_m', 0.)),
                    height_calibration=public.get('height_calibration'), head_calibration=None,
                    policy=policy_for(entry['dataset']) + ('-nymeria-sparse-nohand-window-v2' if entry['dataset']=='Nymeria' else ''))
                kind = 'body'
            key = identity(context)
            candidate = dict(id=sid, title=entry['title'], work=str(work), kind=kind,
                source_work=str(source_work) if source_work else None)
            if key not in groups:
                groups[key] = dict(asset_id=key, context=context, entries=[], selected=candidate,
                                   status='queued', created_at=time.time())
            group = groups[key]; group['entries'].append(candidate)
            if (work/'prior-ready.json').exists():
                verified(work); group['selected'] = candidate; group['status'] = 'existing'
        except Exception as error:
            failed.append(dict(id=sid, error=type(error).__name__+': '+str(error)))
    return dict(schema=SCHEMA, label='Human motion assets', catalog_entries=len(catalog),
        excluded=excluded, planning_errors=failed, assets=list(groups.values()),
        code_dir=str(Path(__file__).parent), created_at=time.time())


def audit(work, expected_frames):
    import numpy as np
    receipt = verified(work)
    with np.load(motion_path(work)) as data:
        joints = data['joints']; times = data['timestamps_s']
        if len(joints) != expected_frames or len(times) != expected_frames:
            raise ValueError('Human-motion frame count differs from source')
        for key in ['joints', 'vertices', 'global_rotation', 'timestamps_s']:
            if not np.isfinite(data[key]).all():
                raise ValueError('Non-finite human-motion artifact: '+key)
        if len(times)>1 and not np.allclose(np.diff(times), 1/30, atol=1e-5):
            raise ValueError('Human-motion timestamps are not 30 Hz')
        low = data['vertices'][:,:,2].min(1)
        pelvis = np.percentile(joints[:,0,2], [0,50,100]).tolist()
    with np.load(Path(work)/'observations.npz') as obs:
        floor = float(obs['floor_z'])
    report = dict(frames=expected_frames, fps=30, finite=True, raw_sha256=receipt['raw_sha256'],
        pelvis_z_m_min_median_max=pelvis, floor_z_m=floor,
        mesh_minimum_z_minus_declared_floor_m=float(low.min()-floor),
        reconstructed=True, ground_truth=False, physical_validation=False,
        fingers='Observed source hand trajectories remain separate from body-prior fingers')
    path = Path(work)/'human_validation.json'
    if path.exists():report['body_transfer_audit']=json.loads(path.read_text())
    return report


def aggregate(manifest):
    counts = {}
    for asset in manifest['assets']:counts[asset['status']]=counts.get(asset['status'],0)+1
    return dict(schema=SCHEMA, label='Human motion assets', counts=counts,
        total_unique_assets=len(manifest['assets']), catalog_entries=manifest['catalog_entries'],
        excluded_umi=len(manifest['excluded']), planning_errors=manifest['planning_errors'],
        updated_at=time.time(), single_physical_gpu=3)


def run(manifest_path, gpu):
    if gpu != 3:raise ValueError('This isolated batch is restricted to physical GPU 3')
    from arenalab_workbench.gpu_scheduler import resource_lock, run as gpu_run
    manifest_path=Path(manifest_path); manifest=json.loads(manifest_path.read_text())
    base=manifest_path.parent; library=base.parent.parent
    # A durable lock prevents two resume invocations building the same batch.
    with resource_lock('human-library:'+str(manifest_path.resolve())):
        save(base/'process.json',dict(pid=os.getpid(),status='running',gpu=gpu,started_at=time.time()))
        for asset in manifest['assets']:
            row=asset['selected']; work=Path(row['work']); sid=row['id']
            if asset['status']=='failed':continue
            try:
                with resource_lock('source:'+sid):
                    if not (work/'prior-ready.json').exists():
                        asset.update(status='queued',queued_at=time.time());save(manifest_path,manifest)
                        save(base/'progress.json',aggregate(manifest))
                        command=[sys.executable,str(Path(__file__).resolve()),'infer',
                                 '--manifest',str(manifest_path),'--asset',asset['asset_id']]
                        env=dict(os.environ,ARENA_GPU_POOL=str(gpu),CUDA_VISIBLE_DEVICES=str(gpu),
                            EGL_DEVICE_ID=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                            PYTHONPATH=paths.pythonpath())
                        with (base/(asset['asset_id']+'.log')).open('a') as log:
                            gpu_run(command,label='Human motion asset: '+row['title'],env=env,
                                    stdout=log,stderr=subprocess.STDOUT,check=True)
                    report=audit(work,asset['context']['frames'])
                    if abs(report['floor_z_m']-asset['context']['floor_z_m'])>1e-9:
                        raise ValueError('Saved body floor differs from requested height; regenerate a version')
                    descriptor=dict(schema=SCHEMA,asset_id=asset['asset_id'],
                        label=family_for_work(work).upper()+' human motion',input=asset['context'],work=str(work),
                        aliases=[r['id'] for r in asset['entries']],audit=report,
                        paths={name:str(work/name) for name in report['raw_sha256']},
                        saved_at=time.time(),robot_independent=True)
                    save(library/'assets'/asset['asset_id']/'asset.json',descriptor)
                    asset.update(status='ready',completed_at=time.time(),descriptor=str(library/'assets'/asset['asset_id']/'asset.json'))
                    # Index is separate from the robot-result cache and library catalog.
                    with resource_lock('human-library-index'):
                        index_path=library/'index.json'
                        index=json.loads(index_path.read_text()) if index_path.exists() else dict(schema=SCHEMA,label='Human motion assets',assets={})
                        index['assets'][asset['asset_id']]=descriptor;save(index_path,index)
            except Exception as error:
                asset.update(status='failed',error=type(error).__name__+': '+str(error),failed_at=time.time())
            save(manifest_path,manifest);save(base/'progress.json',aggregate(manifest))
            print(json.dumps(dict(id=sid,status=asset['status'],error=asset.get('error'))),flush=True)
        summary=aggregate(manifest)
        summary['status']='completed_with_errors' if summary['planning_errors'] or summary['counts'].get('failed') else 'completed'
        save(base/'progress.json',summary);save(base/'process.json',dict(pid=os.getpid(),status=summary['status'],finished_at=time.time(),gpu=gpu))


def infer(manifest_path, asset_id):
    manifest=json.loads(Path(manifest_path).read_text())
    asset=next(x for x in manifest['assets'] if x['asset_id']==asset_id);row=asset['selected']
    asset.update(status='running',started_at=time.time())
    # This child owns the GPU lease; it may safely update the one sequential job.
    save(manifest_path,manifest);save(Path(manifest_path).parent/'progress.json',aggregate(manifest))
    if row['kind']=='gripper':
        from arenalab_workbench.gripper_body import run as recover
        actual=recover(row['source_work'])
    else:
        from arenalab_workbench.body_prior import run as recover
        actual=recover(row['id'],Path(manifest_path).parent/(asset_id+'-body-stage.json'))
    if Path(actual).resolve()!=Path(row['work']).resolve():
        raise ValueError('Body asset location changed after planning; retain outputs and re-plan')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run','infer'])
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--asset')
    p.add_argument('--gpu',type=int,default=3)
    a=p.parse_args()
    if a.mode=='plan':
        result=plan();save(a.manifest,result);save(a.manifest.parent/'progress.json',aggregate(result));print(json.dumps(aggregate(result)),flush=True)
    elif a.mode=='infer':infer(a.manifest,a.asset)
    else:run(a.manifest,a.gpu)
