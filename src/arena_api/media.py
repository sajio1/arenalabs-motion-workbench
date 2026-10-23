"""Versioned delivery of public viewer assets with private-workspace checks."""
import hashlib
import json
import secrets
from pathlib import Path
from fastapi import HTTPException, Request, Response
from fastapi.responses import FileResponse


def attach_media(app, storage):
    root = Path(storage).resolve()

    @app.head('/media/{relative:path}', include_in_schema=False)
    @app.get('/media/{relative:path}', tags=['results'])
    def media(relative: str, request: Request):
        parts = relative.split('/')
        if (not relative or '\\' in relative or ':' in relative or relative.startswith('/')
                or any(p.startswith('.') for p in parts)
                or parts[0] in {'inputs', 'source-recordings'}
                or relative in {'uploaded/registry.json', 'viewer-conversions.json'}):
            raise HTTPException(404, 'Asset unavailable')
        path = (root / relative).resolve()
        allowed = {'.json', '.glb', '.gltf', '.bin', '.gz', '.mp4', '.webm', '.jpg', '.jpeg', '.png', '.webp', '.svg', '.ktx2'}
        if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in allowed:
            raise HTTPException(404, 'Asset unavailable')
        if parts[0] == 'jobs' and (len(parts) != 3 or parts[-1] != 'playback.json'):
            raise HTTPException(404, 'Asset unavailable')
        token = request.cookies.get('arena_workspace')
        who = hashlib.sha256(token.encode()).hexdigest() if token else ''
        private = False
        for parent in path.parents:
            marker = parent / '.workspace.json'
            if marker.is_file():
                private = True
                expected = json.loads(marker.read_text()).get('owner', '')
                if not who or not secrets.compare_digest(expected, who):
                    raise HTTPException(404, 'Asset unavailable')
            if parent == root: break
        response = FileResponse(path, headers={'Cache-Control': 'private, no-store' if private
                                 else 'private, max-age=0, must-revalidate', 'Vary': 'Cookie'})
        response.set_stat_headers(path.stat())
        if request.headers.get('if-none-match') == response.headers['etag']:
            return Response(status_code=304, headers={key: response.headers[key] for key in ('etag', 'cache-control', 'vary')})
        return response
