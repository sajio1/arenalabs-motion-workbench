"""Operator catalog, original cameras, motion, geometry and existing recordings."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlsplit
from fastapi import HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

PREFIX = '/api/v1'
COLLECTIONS = {'default': 'catalog.json', 'single-v29': 'single-v29-candidates.json',
               'kitchen-v29b': 'kitchen-v29b-candidates.json', 'scene-v30': 'scene-v30-candidates.json',
               'source-demo': 'source-demo-candidates.json', '1': 'repair-candidates.json'}
KINDS = {'source', 'poster', 'meta', 'motion', 'environment', 'task_report',
         'blender', 'recording', 'camera_head', 'camera_left', 'camera_right'}


class OperatorService:
    def __init__(self, root): self.root = Path(root).resolve() if root else None

    def path(self, reference):
        if self.root is None: raise HTTPException(503, 'Operator storage not configured')
        if not isinstance(reference, str): raise HTTPException(404, 'Asset unavailable')
        parsed = urlsplit(reference)
        if (parsed.scheme or parsed.netloc or reference.startswith('/') or '\\' in reference
                or ':' in reference or any(p.startswith('.') for p in reference.split('/'))):
            raise HTTPException(409, 'Invalid Operator asset reference')
        path = (self.root / reference).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise HTTPException(404, 'Asset unavailable')
        return path

    def catalog(self, collection='default'):
        if collection not in COLLECTIONS: raise HTTPException(404, 'Unknown Operator collection')
        try: value = json.loads(self.path('library/' + COLLECTIONS[collection]).read_text())
        except HTTPException as exc:
            if exc.status_code == 404: raise HTTPException(404, 'Operator collection unavailable') from exc
            raise
        return value if isinstance(value, dict) else {'cases': value}

    def rows(self, collection='default'):
        value = self.catalog(collection)
        return value.get('cases', value.get('episodes', value.get('items', [])))

    def row(self, ident, collection='default'):
        value = next((r for r in self.rows(collection) if r.get('id') == ident), None)
        if value is None: raise HTTPException(404, 'Episode unavailable')
        return value

    def cameras(self, ident):
        try: return json.loads(self.path('library/real-cameras.json').read_text()).get('views', {}).get(ident, {})
        except HTTPException as exc:
            if exc.status_code == 404: return {}
            raise

    def reference(self, row, kind):
        if kind.startswith('camera_'):
            value = (self.cameras(row['id']).get(kind.removeprefix('camera_')) or {}).get('url')
            # Recorded camera catalog uses either relative URLs or /isaac/ URLs.
            return value.removeprefix('/isaac/') if isinstance(value, str) else None
        if kind == 'blender':
            if row.get('scene_asset_kind') == 'blender_static_reuse':
                return 'cache/curated50/' + row['family'] + '_static.blend'
            if row.get('scene_asset_kind') in {'blender_contact_repair', 'blender_preview', 'blender_source_demo'}:
                return (row.get('environment') or '').removesuffix('.glb') + '.blend'
            return None
        if kind == 'recording': return 'recording/' + row['id'] if row.get('recording') else None
        return row.get(kind)

    def asset_url(self, row, kind, collection):
        reference = self.reference(row, kind)
        if not reference: return None
        # Include source revision in URL so browser caches cannot mask updated data.
        try:
            path = self.path(reference); stat = path.stat()
            rev = hashlib.sha256(f'{reference}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:20]
        except HTTPException: rev = 'unavailable'
        return (PREFIX + '/simulation-episodes/' + quote(row['id'], safe='') + '/assets/' + kind
                + '?collection=' + quote(collection) + '&revision=' + rev)

    def public(self, row, collection):
        result = dict(row)
        for kind in KINDS:
            url = self.asset_url(row, kind, collection)
            if url: result[kind + '_url' if kind in {'blender', 'recording'} else kind] = url
        result['cameras'] = {key: {**spec, 'url': self.asset_url(row, 'camera_' + key, collection)}
                             for key, spec in self.cameras(row['id']).items() if spec and key in {'head', 'left', 'right'}}
        return result


def attach_operator(app, root):
    service = OperatorService(root)
    app.state.operator = service

    @app.get('/simulation-episodes', tags=['operator'])
    def episodes(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), collection: str = 'default'):
        rows = service.rows(collection)
        return {'items': [service.public(r, collection) for r in rows[offset:offset+limit]], 'total': len(rows),
                'offset': offset, 'limit': limit, 'collection': collection, 'mode': 'recorded_replay', 'live_physics_rollout': False}

    @app.get('/operator/resources/{kind}', tags=['operator'])
    def robot_resource(kind: str):
        reference = {'robot-metadata': 'cache/robot-v2.json', 'robot-geometry': 'cache/robot-v2.bin.gz'}.get(kind)
        return FileResponse(service.path(reference), headers={'Cache-Control': 'private, max-age=0, must-revalidate'})

    @app.get('/simulation-episodes/{ident}', tags=['operator'])
    def episode(ident: str, collection: str = 'default'):
        return {'episode': service.public(service.row(ident, collection), collection), 'mode': 'recorded_replay',
                'live_physics_rollout': False, 'evidence_semantics': 'Preserved source reports, including unverified or conflicting labels.'}

    @app.get('/simulation-episodes/{ident}/playback', tags=['operator'])
    def playback(ident: str, collection: str = 'default'):
        row = service.row(ident, collection)
        return {'episode_id': ident, 'assets': {k: url for k in KINDS if (url := service.asset_url(row, k, collection))},
                'mode': 'recorded_replay', 'source_metadata': service.public(row, collection)}

    @app.head('/simulation-episodes/{ident}/assets/{kind}', include_in_schema=False)
    @app.get('/simulation-episodes/{ident}/assets/{kind}', tags=['operator'])
    def asset(ident: str, kind: str, request: Request, collection: str = 'default', revision: str = ''):
        if kind not in KINDS: raise HTTPException(404, 'Asset unavailable')
        row = service.row(ident, collection)
        path = service.path(service.reference(row, kind))
        current = service.asset_url(row, kind, collection).rsplit('revision=', 1)[-1]
        if revision and revision != current: raise HTTPException(409, 'Episode changed; reload the catalog')
        response = FileResponse(path, media_type='video/mp4' if kind == 'recording' else None,
                                headers={'Cache-Control': 'private, max-age=300' if revision else 'private, max-age=0, must-revalidate'})
        response.set_stat_headers(path.stat())
        if request.headers.get('if-none-match') == response.headers['etag']:
            return Response(status_code=304, headers={'ETag': response.headers['etag']})
        return response
