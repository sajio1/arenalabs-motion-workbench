"""Distribute independent SAM categories; each keeps full temporal propagation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib,json,os,shutil,subprocess,sys,time
from gpu_scheduler import run,write,resource_lock

def main():
    env=os.environ.copy()
    if env.get('ARENA_GPU_LEASE'):
        raise RuntimeError('Category coordinator must not hold a GPU lease')
    prompts=json.loads(env['ARENA_SEGMENT_PROMPTS'])
    output=Path(env['ARENA_SEGMENT_OUTPUT']);output.mkdir(parents=True,exist_ok=True)
    worker=Path(env.get('ARENA_SEGMENT_SCRIPT',Path(__file__).with_name('segment_scene.py')))
    groups=[prompts] if len(env.get('ARENA_GPU_POOL','0,1,2,3').split(','))==1 else [prompts[i::4] for i in range(min(4,len(prompts)))]
    frame_hash=hashlib.sha256()
    for p in sorted(Path(env['ARENA_SEGMENT_FRAMES']).glob('*.jpg')):frame_hash.update(p.name.encode()+p.read_bytes())
    frame_identity=frame_hash.hexdigest()
    started=time.monotonic()
    helpers=[worker.with_name(name) for name in ['tracking_recovery.py','sam_runtime.py','sam_features.py']]
    worker_hash=hashlib.sha256(worker.read_bytes()+b''.join(p.read_bytes() for p in helpers if p.exists())).hexdigest()
    weights=Path('/home/ubuntu/arenalab-experiments/manipulation4d/sam31-weights/sam3.1_multiplex.pt')
    stat=weights.stat() if weights.exists() else None
    model_identity=(str(weights),stat.st_size,stat.st_mtime_ns) if stat else None
    cache=Path(env.get('ARENA_SAM_CATEGORY_CACHE','/home/ubuntu/arenalab-experiments/manipulation4d/sam-category-cache'))
    def identity(item):
        return hashlib.sha256(json.dumps([frame_identity,worker_hash,model_identity,item[1:]],sort_keys=True).encode()).hexdigest()
    def infer(group):
        keys={item[0]:identity(item) for item in group}
        key=hashlib.sha256(json.dumps([frame_identity,worker_hash,group],sort_keys=True).encode()).hexdigest()
        directory=output/'parallel'/key;directory.mkdir(parents=True,exist_ok=True)
        details={'prompts':{}};hits=[]
        # Acquire in global order before leasing a GPU; concurrent requests for
        # identical pixels/categories share one inference and cannot deadlock.
        with ExitStack() as locks:
            for digest in sorted(set(keys.values())):locks.enter_context(resource_lock('sam-category:'+str(cache)+':'+digest))
            missing=[]
            for item in group:
                saved=cache/keys[item[0]]
                if (saved/'complete.json').exists() and (saved/'masks.npz').exists():
                    shutil.copyfile(saved/'masks.npz',directory/(item[0]+'-masks.npz'))
                    detail=json.loads((saved/'complete.json').read_text())
                    detail.update(cache_hit=True,cached_inference_seconds=detail.get('seconds'),seconds=0.)
                    details['prompts'][item[0]]=detail;hits.append(item[0])
                else:missing.append(item)
            if missing:
                child=env.copy();child.update(ARENA_SEGMENT_OUTPUT=str(directory),ARENA_SEGMENT_PROMPTS=json.dumps(missing))
                with (directory/'worker.log').open('w') as log:
                    run([sys.executable,str(worker)],env=child,label='SAM: '+', '.join(p[1] for p in missing),stdout=log,stderr=subprocess.STDOUT,check=True)
                inferred=json.loads((directory/'segmentation.json').read_text())
                details.update({k:v for k,v in inferred.items() if k!='prompts'})
                for item in missing:
                    source=directory/(item[0]+'-masks.npz')
                    if not source.exists():raise RuntimeError('SAM category output missing')
                    saved=cache/keys[item[0]];saved.mkdir(parents=True,exist_ok=True)
                    temporary=saved/'masks.npz.tmp';shutil.copyfile(source,temporary);temporary.replace(saved/'masks.npz')
                    detail=inferred['prompts'][item[0]];detail['cache_hit']=False
                    write(saved/'complete.json',detail);details['prompts'][item[0]]=detail
            write(directory/'segmentation.json',details)
            write(directory/'complete.json',{'prompts':group,'cache_hits':hits})
        return directory,group
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(infer,groups))
    report=json.loads((output/'segmentation.json').read_text()) if (output/'segmentation.json').exists() else {'model':'SAM 3.1 Object Multiplex','source_fps':30,'mask_fps':15,'prompts':{}}
    for directory,group in results:
        details=json.loads((directory/'segmentation.json').read_text())
        for item in group:
            name=item[0]+'-masks.npz';temporary=output/(name+'.tmp')
            shutil.copyfile(directory/name,temporary);temporary.replace(output/name)
            report['prompts'][item[0]]=details['prompts'][item[0]]
    report.update(frame_identity=frame_identity,seconds=time.monotonic()-started,parallel_category_groups=len(groups),temporal_sequences_split=False)
    write(output/'segmentation.json',report)
    print('SEGMENTATION_COMPLETE',flush=True)

if __name__=='__main__':main()
