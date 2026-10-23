"""CLI for the same HTTP API. Workspace tokens are supplied via the environment."""
import argparse,json,os
from .client import ArenaClient

def main():
    parser=argparse.ArgumentParser(description='ArenaLabs current platform API')
    parser.add_argument('--url',default=os.environ.get('ARENA_API_URL','http://127.0.0.1:8765'))
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['capabilities','robots','end-effectors','episodes']:sub.add_parser(name)
    p=sub.add_parser('recordings');p.add_argument('--workflow',default='manipulation');p.add_argument('--limit',type=int,default=50)
    p=sub.add_parser('job');p.add_argument('id')
    for name in ['plan','run']:
        p=sub.add_parser(name);p.add_argument('--workflow',choices=['manipulation','locomotion','export'],required=True)
        p.add_argument('--recording',required=True);p.add_argument('--robot',required=True);p.add_argument('--hand')
        if name=='run':p.add_argument('--key',required=True)
    args=parser.parse_args()
    with ArenaClient(args.url,workspace_token=os.environ.get('ARENA_WORKSPACE_TOKEN')) as client:
        if args.command in {'run','plan','job'} and not os.environ.get('ARENA_WORKSPACE_TOKEN'):
            parser.error('Set ARENA_WORKSPACE_TOKEN to a retained workspace cookie for job access')
        if args.command in {'plan','run'}:
            spec=dict(workflow=args.workflow,recording_id=args.recording,robot_id=args.robot)
            if args.hand is not None:spec['hand_id']=args.hand
            result=client.plan(**spec) if args.command=='plan' else client.run(idempotency_key=args.key,**spec)
        elif args.command=='recordings':result=client.recordings(args.workflow,limit=args.limit)
        elif args.command=='job':result=client.job(args.id)
        else:result=getattr(client,args.command.replace('-','_'))()
        print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
