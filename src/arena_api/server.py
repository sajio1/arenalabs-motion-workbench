"""Additive production launcher; existing algorithm packages need no replacement."""
import argparse
import os
from .app import create_api

def attach(app):
    if any(getattr(r,'path',None)=='/api/v1' for r in app.routes):return app
    from arenalab_workbench.settings import paths
    workbench=next(r.app for r in app.routes if getattr(r,'path',None)=='/api/4d')
    app.mount('/api/v1',create_api(app,workbench,storage=paths.store,
              database=os.environ.get('ARENA_API_DATABASE',str(paths.workbench/'api-v1/registry.sqlite3')),
              operator_root=os.environ.get('ARENA_OPERATOR_ROOT','/var/www/openarm-viewer')))
    return app

def main():
    import uvicorn
    from arenalab_workbench.serve_workbench import create_app
    from arenalab_workbench.settings import paths
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(paths.viewer_config))
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    uvicorn.run(attach(create_app(args.config)),host=args.host,port=args.port)

if __name__=='__main__':main()
