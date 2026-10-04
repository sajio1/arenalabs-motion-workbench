"""SOMA/SMPL/BVH/canonical manifests share the existing 100+100 solver."""
from pathlib import Path
import os,subprocess,sys
from greenwich_pipelines.config import ROOT
from greenwich_pipelines.artifacts import artifact,write_json,inspect_native

def run(job,robots):
    from greenwich_motion_sdk.source_batch import read_sources
    sources=read_sources(job.source)
    duration=job.max_seconds or max(float(r['_clip'].timestamps_s[-1]-r['_clip'].timestamps_s[0]+1/30) for r in sources)
    roster=job.output/'robots.resolved.json';write_json(roster,robots)
    command=[sys.executable,'-m','greenwich_pipelines.pipelines.fullbody.backend','--repo',str(ROOT),'--pipeline',str(ROOT),'--source-manifest',str(job.source),'--robots',str(roster),'--output',str(job.output/'native'),'--fps','30','--max-seconds',str(duration),'--model','prime','--run-dir',os.environ['GREENWICH_WEIGHTS'],'--contact-config',str(ROOT/'configs/policies/fullbody.json'),'--ffmpeg','ffmpeg','--skip-preview']
    subprocess.run(command,check=True)
    results=[]
    for path in sorted((job.output/'native').rglob('motion.npz')):
        frames=inspect_native(path,'fullbody')
        results.append(artifact(job.output,path,'motion',frames=frames,coordinate_frame='right-handed-y-up',position_unit='cm',joint_encoding='descriptor-angle-slots',robot=path.parent.name,clip=path.parent.parent.name))
        prior=path.with_name('prime_original.npz')
        results.append(artifact(job.output,prior,'model_prior',robot=path.parent.name,clip=path.parent.parent.name))
        if path.with_name('report.json').exists():results.append(artifact(job.output,path.with_name('report.json'),'report'))
    if not results:raise RuntimeError('Backend produced no motion artifacts')
    return results
