"""One local entry point. Run each backend in its own isolated process."""
import argparse,importlib.util,json,subprocess,sys
from pathlib import Path
from .config import Job,PIPELINES,robot_rows,read_json,resolve

def main():
    parser=argparse.ArgumentParser(prog='greenwich-pipelines')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list')
    for name in ('plan','doctor','run'):
        p=sub.add_parser(name);p.add_argument('job',type=Path)
    args=parser.parse_args()
    if args.command=='list':print(json.dumps({'pipelines':PIPELINES,'fps':30},indent=2));return
    try:
        job=Job.load(args.job);rows=robot_rows(job.robots)
        if args.command=='plan':
            print(json.dumps(dict(id=job.id,pipeline=job.pipeline,source=str(job.source),robots=[r['body'] for r in rows],output=str(job.output),fps=30,acceptance=job.acceptance,weights=read_json(job.runtime).get('weights')),indent=2));return
        if args.command=='doctor':
            checks={k:importlib.util.find_spec(k) is not None for k in ['numpy','scipy','torch','mujoco','transformers','safetensors','platformdirs']+(['h5py','trimesh'] if job.pipeline=='gento' else [])}
            checks.update(source=job.source.is_file(),runtime=job.runtime.is_file(),output_unused=not job.output.exists())
            cfg=read_json(job.runtime);weights=resolve(job.runtime.parent,cfg['weights'])
            checks['checkpoint']=(weights/'config.json').is_file() and (weights/'model.safetensors').is_file()
            for row in rows:
                checks['robot:'+row['body']]=Path(row.get('metadata') or row.get('model') or row.get('xml')).is_file()
            print(json.dumps(checks,indent=2))
            if not all(checks.values()):raise SystemExit(1)
            return
        subprocess.run([sys.executable,'-m','greenwich_pipelines.worker',str(job.path)],check=True)
    except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as exc:parser.exit(2,f'{exc}\n')

if __name__=='__main__':main()
