"""Bindings to the running application services, with no internal HTTP requests.

Existing handlers own their locks, workers and validation. Binding those exact
instances lets both API generations share one queue and one numerical policy.
Only explicit operations are callable; arbitrary route dispatch is not exposed.
"""
import inspect
from fastapi import HTTPException, Response
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool


class Operation:
    def __init__(self, app, path, method='GET'):
        self.route = next((r for r in app.routes if isinstance(r, APIRoute)
                           and r.path == path and method in r.methods), None)
        if self.route is None:
            raise RuntimeError(f'Missing application operation: {method} {path}')
        dep = self.route.dependant
        # Fail at startup if a future implementation adds dependencies that this
        # service binding cannot supply. Never silently bypass authentication.
        if dep.dependencies or dep.query_params or dep.header_params or dep.cookie_params:
            raise RuntimeError(f'Operation requires a new explicit binding: {path}')

    async def __call__(self, request, payload=None, **path):
        dep = self.route.dependant
        values = dict(path)
        if dep.request_param_name: values[dep.request_param_name] = request
        if dep.response_param_name: values[dep.response_param_name] = Response()
        for field in dep.body_params:
            value, errors = field.validate(payload, {}, loc=('body',))
            if errors: raise HTTPException(422, 'Invalid operation input')
            values[field.name] = value
        endpoint = self.route.endpoint
        if inspect.iscoroutinefunction(endpoint): return await endpoint(**values)
        return await run_in_threadpool(endpoint, **values)


class PlatformServices:
    def __init__(self, viewer, workbench):
        self.viewer, self.workbench = viewer, workbench
        self.library = Operation(workbench, '/library')
        self.robots = Operation(workbench, '/robots')
        self.catalog = Operation(viewer, '/api/catalog/{page}')
        self.generate = Operation(workbench, '/generate', 'POST')
        self.convert = Operation(viewer, '/api/convert', 'POST')
        self.export = Operation(workbench, '/exports', 'POST')
        self.manipulation_status = self._status(workbench, '/jobs/')
        self.locomotion_status = self._status(viewer, '/api/jobs/')
        self.export_status = self._status(workbench, '/exports/')
        self.export_download = None
        if any(getattr(r, 'path', None) == '/exports/{ident}/download' for r in workbench.routes):
            self.export_download = Operation(workbench, '/exports/{ident}/download')

    @staticmethod
    def _status(app, prefix):
        route = next(r for r in app.routes if isinstance(r, APIRoute)
                     and r.path.startswith(prefix) and r.path.count('/') == prefix.count('/')
                     and 'GET' in r.methods)
        return Operation(app, route.path)

    async def recordings(self, request, workflow):
        if workflow == 'locomotion': return (await self.catalog(request, page='locomotion'))['clips']
        return await self.library(request)

    async def submit(self, workflow, request, payload):
        return await {'manipulation': self.generate, 'locomotion': self.convert,
                      'export': self.export}[workflow](request, payload)

    async def input_requirements(self, scene_id, request):
        operation = getattr(self.workbench.state, 'input_requirements', None)
        if operation is None: return None
        return await run_in_threadpool(operation, scene_id, request)

    async def status(self, workflow, ident, request):
        op = getattr(self, workflow + '_status')
        key = op.route.dependant.path_params[0].name
        return await op(request, **{key: ident})

    async def download(self, ident, request):
        if self.export_download is None: raise HTTPException(503, 'Export download unavailable')
        return await self.export_download(request, ident=ident)
