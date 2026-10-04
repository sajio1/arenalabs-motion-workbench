"""Transaction boundary for one conversion; partial results remain diagnosable."""
import importlib,sys,traceback
from pathlib import Path
from .config import Job,robot_rows
from .runtime import configure
from .artifacts import write_json,sha256

def run(job):
    rows=robot_rows(job.robots)
    if not job.source.is_file():raise FileNotFoundError(job.source)
    configure(job.runtime)
    job.output.mkdir(parents=True,exist_ok=False)
    status=job.output/'status.json'
    inputs={str(p):sha256(p) for p in [job.path,job.source,job.robots,job.runtime]}
    write_json(status,dict(state='running',job=job.id,pipeline=job.pipeline))
    try:
        adapter=importlib.import_module('greenwich_pipelines.pipelines.'+job.pipeline+'.runner')
        artifacts=adapter.run(job,rows)
        if any(sha256(p)!=digest for p,digest in inputs.items()):raise RuntimeError('Input changed during conversion')
        manifest=dict(schema='greenwich.result.v1',job=job.id,pipeline=job.pipeline,fps=30,artifacts=artifacts,input_sha256=inputs,accepted=None,acceptance_module=None,quality_status='not_evaluated')
        write_json(job.output/'result.json',manifest)
        if job.acceptance:
            module,fn=job.acceptance.split(':',1)
            result=getattr(importlib.import_module(module),fn)(job.output/'result.json')
            if not isinstance(result,dict) or type(result.get('accepted')) is not bool:raise ValueError('Acceptance module must return {accepted: bool, ...}')
            if any(sha256(job.output/a['path'])!=a['sha256'] for a in artifacts):raise RuntimeError('Acceptance module changed numerical artifacts')
            write_json(job.output/'acceptance.json',result)
            manifest.update(accepted=result['accepted'],acceptance_module=job.acceptance,quality_status='passed' if result['accepted'] else 'failed')
            write_json(job.output/'result.json',manifest)
        write_json(status,dict(state='completed',job=job.id,pipeline=job.pipeline,quality_status=manifest['quality_status']))
        return manifest
    except Exception as exc:
        write_json(status,dict(state='failed',job=job.id,error=type(exc).__name__,message=str(exc)))
        raise

if __name__=='__main__':run(Job.load(Path(sys.argv[1])))
