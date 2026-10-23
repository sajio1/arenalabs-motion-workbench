"""Administrator-owned local source rigs, separate from selectable target robots.

Requests select only a registered recording ID. Neither its source descriptor nor
this registry is accepted as a request payload; private models stay at the local
decoding endpoint. Existing target rigs remain valid source rigs for compatibility.
"""
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path

from .settings import paths


def source_rows():
    registry = Path(os.environ.get('ARENA_SOURCE_ROBOT_REGISTRY', str(paths.config/'source-robots.json')))
    if not registry.is_file():
        return {}
    payload = json.loads(registry.read_text())
    rows = payload.get('robots') if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError('Local source robot registry requires a robots list')
    result = {}
    for entry in rows:
        key = entry.get('id')
        row = deepcopy(entry.get('pipeline_row'))
        if not isinstance(key, str) or not key or key in result or not isinstance(row, dict):
            raise ValueError('Local source robot IDs and pipeline rows must be unique and explicit')
        if not isinstance(row.get('xml'), str) or not row['xml']:
            raise ValueError('Local source robot requires an administrator-configured XML')
        for field in ('xml', 'model'):
            if row.get(field):
                value = Path(row[field])
                row[field] = str((value if value.is_absolute() else registry.parent/value).resolve())
        if row.get('model', row['xml']) != row['xml']:
            raise ValueError('Local source robot model and XML must identify the same asset')
        result[key] = row
    return result


def resolve_source_robot(identifier):
    """Return a private pipeline row without adding it to the target roster."""
    if not isinstance(identifier, str) or not identifier:
        raise ValueError('Complete robot motion requires a locally registered source robot_id')
    row = source_rows().get(identifier)
    if row is None:
        from .robot_registry import roster
        target = roster().get(identifier)
        row = target.get('pipeline_row') if target else None
    if row is None:
        raise ValueError('Source robot is unavailable at this decoding endpoint: ' + identifier)
    return deepcopy(row)


@lru_cache(maxsize=16)
def _validated(row_json, asset_identity):
    from arenalab_motion.robot_metadata import ensure_robot_metadata
    return ensure_robot_metadata(json.loads(row_json))


def validated_source_robot(identifier):
    """Reuse fixed rig validation across a corpus, invalidating on asset changes."""
    from .motion_backend import robot_asset_identity
    row = resolve_source_robot(identifier)
    assets = robot_asset_identity(row['xml'])
    return deepcopy(_validated(json.dumps(row,sort_keys=True), json.dumps(assets,sort_keys=True)))
