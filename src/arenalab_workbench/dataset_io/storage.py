"""Filesystem boundaries and browser workspace ownership, shared by API and workers."""
from arenalab_workbench.settings import paths
from pathlib import Path
import hashlib
import json
import os
import secrets

ROOT = Path(os.environ.get('ARENA_WORKBENCH_ROOT', str(paths.workbench)))
STORE = Path(os.environ.get('ARENA_STORE', str(paths.store)))
COOKIE = 'arena_workspace'


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + secrets.token_hex(6) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def within(root, relative):
    """Reject absolute paths, traversal and symlink escapes on every platform."""
    relative = str(relative).replace('\\', '/')
    if relative.startswith('/') or ':' in relative or '..' in relative.split('/'):
        raise ValueError('Package path must stay inside the recording directory')
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Package path escapes its recording directory')
    return result


def owner(token):
    return hashlib.sha256(token.encode()).hexdigest() if token else None


def uploads():
    return read(ROOT / 'user-data/registry.json', {})


def entry(scene_id):
    return uploads().get(scene_id)


def mark_private(path, workspace):
    if workspace:
        write(Path(path) / '.workspace.json', {'owner': workspace})


def permitted(path, token):
    path = Path(path).resolve()
    if not path.is_relative_to(STORE.resolve()):
        return False
    for directory in [path.parent, *path.parents]:
        marker = read(directory / '.workspace.json')
        if marker:
            return bool(token) and secrets.compare_digest(marker['owner'], owner(token))
        if directory == STORE.resolve():
            break
    return True


def authorize_scene(scene_id, token):
    record = entry(scene_id)
    if record is None:
        # Height variants inherit the original public entry's ownership.
        record = read(ROOT / 'height-previews.json', {}).get(scene_id)
    workspace = (record or {}).get('public', {}).get('owner')
    return not workspace or bool(token) and secrets.compare_digest(workspace, owner(token))
