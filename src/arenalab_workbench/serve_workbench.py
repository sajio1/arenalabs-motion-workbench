"""Production HTTP entry point. Importing does not launch jobs or create storage."""
import argparse
from arenalab_workbench.settings import paths


def create_app(config_path=None):
    from greenwich_pipelines.viewer.server import create_app as create_viewer
    from arenalab_workbench.workbench_api import create_api
    from arenalab_workbench.dataset_io.routes import protect_uploaded_assets
    app = create_viewer(config_path or paths.viewer_config)
    app.mount('/api/4d', create_api(config_path))
    protect_uploaded_assets(app)
    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description='ArenaLabs multimodal motion workbench')
    parser.add_argument('--config', default=str(paths.viewer_config))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
