"""Publish existing typed resource handlers under the versioned API."""

def alias(app, source, old_path, new_path, method, tag):
    """Reuse typed existing handlers, including streaming file responses/uploads."""
    from fastapi.routing import APIRoute
    route = next((r for r in source.routes if isinstance(r,APIRoute) and r.path==old_path and method in r.methods),None)
    if route is None:raise RuntimeError('Missing platform route: '+method+' '+old_path)
    app.add_api_route(new_path,route.endpoint,methods=[method],response_model=route.response_model,
                      status_code=route.status_code,tags=[tag],
                      name='v1_'+tag+'_'+route.name,dependencies=route.dependencies,
                      description='Adapter for the current '+method+' '+old_path+' operation. Existing validation and data semantics are preserved.')
