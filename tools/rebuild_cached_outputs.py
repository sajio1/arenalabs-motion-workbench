"""Explicit maintenance command; rebuild only pairs in an operator-supplied plan.

Each plan row contains scene_id/robot_id/hand_id, or clip_id/robot_id for
Locomotion. Old source data and inference outputs are never deleted by this tool.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--server',default='http://127.0.0.1:8765')
    parser.add_argument('--workers',type=int,default=4,choices=range(1,17))
    args=parser.parse_args();plan=json.loads(args.plan.read_text());results={};lock=threading.Lock()
    def get(path,body=None):
        request=urllib.request.Request(args.server.rstrip('/')+path,data=json.dumps(body).encode() if body else None,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)
    def save():
        args.report.parent.mkdir(parents=True,exist_ok=True)
        tmp=args.report.with_suffix('.tmp');tmp.write_text(json.dumps(list(results.values()),indent=2));tmp.replace(args.report)
    def run(pair):
        key=json.dumps(pair,sort_keys=True);four_d='scene_id' in pair
        try:
            job=get('/api/4d/generate' if four_d else '/api/convert',pair)
            while job['status'] in {'queued','running'}:
                with lock:results[key]={'request':pair,**job};save()
                time.sleep(3)
                job=get(('/api/4d/jobs/' if four_d else '/api/jobs/')+job['id'])
            if job['status']!='completed':raise RuntimeError(job.get('error') or 'Generation did not complete')
            if four_d:
                scene=get(job['scene_url']);robot=next(r for r in scene['robots'] if r['id']==pair['robot_id']);policy=robot.get('encoder_policy')
            else:policy=get(job['url']).get('encoder_policy')
            if policy!='human-six-pelvis-head-hands-feet-v1':raise RuntimeError('Result does not declare the six-point encoder policy')
            with lock:results[key]={'request':pair,**job,'verified_encoder_policy':policy};save()
        except Exception as error:
            with lock:results[key]={'request':pair,'status':'failed','error':str(error)};save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run,plan))
    if any(r['status']!='completed' for r in results.values()):raise SystemExit(1)


if __name__=='__main__':main()
