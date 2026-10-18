"""Remove a demo embodiment and its derived caches, never the shared human prior."""
from pathlib import Path
import json
import re
import shutil
import time
import uuid


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def delete_robot_assets(config_path, robot_id, *, allow_unregistered=False):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', robot_id):
        raise ValueError('Invalid robot identifier')
    config_path = Path(config_path).resolve()
    config = read(config_path, {})
    storage = Path(config['storage']).resolve()
    registry = storage / 'uploaded/registry.json'
    uploaded = read(registry, [])
    rows = config['robots'] + uploaded
    target = next((r for r in rows if r['id'] == robot_id), None)
    if target is None and not allow_unregistered:
        raise KeyError(robot_id)
    target = target or {'id': robot_id}
    identity = target.get('report', {}).get('selection_key', robot_id)
    removed = {robot_id} | {r['id'] for r in rows
                           if r.get('report', {}).get('selection_key', r['id']) == identity}
    # Do not race a worker that could recreate deleted outputs after cleanup.
    for root in config.get('robot_job_roots', []):
        for job_file in Path(root).glob('*/job.json'):
            job = read(job_file, {})
            state = read(job_file.with_name('state.json'), {})
            if job.get('robot_id') in removed and state.get('status') in {'queued', 'running', 'paused'}:
                raise RuntimeError('This robot is still generating. Delete it after its current task finishes.')

    trash = storage.parent / 'robot-trash' / (str(time.time_ns()) + '-' + robot_id)
    trash.mkdir(parents=True)
    backups = []
    moved = []
    protected_paths = set()

    def save(path, value):
        if path.exists():
            backup = trash / 'metadata' / str(len(backups))
            backup.parent.mkdir(exist_ok=True)
            shutil.copy2(path, backup)
            backups.append({'path': str(path), 'backup': str(backup)})
        write(path, value)

    def retire(path, root=storage):
        path = Path(path).absolute()
        root = Path(root).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError('Cache path is outside its configured root')
        if not path.exists() and not path.is_symlink():
            return
        if any(p == path or p.is_relative_to(path) for p in protected_paths):
            return
        # Never follow symlinks into immutable inputs or shared model assets.
        destination = trash / 'files' / str(len(moved))
        destination.parent.mkdir(exist_ok=True)
        shutil.move(str(path), str(destination))
        moved.append({'path': str(path), 'backup': str(destination)})

    # Native import descriptors remain private so the same ZIP can be added later.
    templates = {r['id']: r for r in config.get('robot_templates', [])}
    for row in config['robots']:
        if row['id'] in removed and row.get('kind') == 'Wheeled':
            templates[row['id']] = row
    config['robot_templates'] = list(templates.values())
    config['robots'] = [r for r in config['robots'] if r['id'] not in removed]
    config['hidden_robot_ids'] = [r for r in config.get('hidden_robot_ids', []) if r not in removed]
    for clip in config.get('clips', []):
        for rid in removed:
            url = clip.get('cache', {}).pop(rid, None)
            if url and url.startswith('/assets/'):
                retire(storage / url.removeprefix('/assets/'))
    for page, presets in config.get('presets', {}).items():
        config['presets'][page] = [p for p in presets if p.get('robot_id') not in removed]
    save(config_path, config)
    save(registry, [r for r in uploaded if r['id'] not in removed])
    conversions_path = storage / 'viewer-conversions.json'
    conversions = read(conversions_path, {})
    for key, value in list(conversions.items()):
        if value.get('robot_id') in removed:
            url = value.get('url', '')
            if url.startswith('/assets/jobs/'):
                retire((storage / url.removeprefix('/assets/')).parent)
            del conversions[key]
    save(conversions_path, conversions)

    manifests = []
    protected_urls = set()
    for path in (storage / 'manipulation4d').rglob('scene.json'):
        scene = read(path, {})
        manifests.append((path, scene))
        for row in scene.get('robots', []):
            if row.get('registry_robot_id', row['id']) not in removed and row['id'] not in removed:
                protected_urls.update(row.get(k) for k in ('model_url', 'playback_url'))
        protected_urls.update(url for rid, url in scene.get('patch_videos', {}).items() if rid not in removed)
    protected_paths.update(storage / url.removeprefix('/assets/') for url in protected_urls
                           if url and url.startswith('/assets/'))
    obsolete_urls = set()
    for path, scene in manifests:
        entries = [r for r in scene.get('robots', [])
                   if r['id'] in removed or r.get('registry_robot_id') in removed]
        ids = removed | {r['id'] for r in entries}
        changed = bool(entries)
        for row in entries:
            obsolete_urls.update(row.get(k) for k in ('model_url', 'playback_url'))
        scene['robots'] = [r for r in scene.get('robots', []) if r not in entries]
        for key in ('patch_videos', 'patch_unavailable', 'patch_kinds', 'hand_motion_audit_urls'):
            for rid in ids:
                if rid in scene.get(key, {}):
                    value = scene[key].pop(rid)
                    if key == 'patch_videos':
                        obsolete_urls.add(value)
                    changed = True
        if scene.get('generation_status', {}).get('robot_id') in ids:
            scene.pop('generation_status', None)
            changed = True
        if changed:
            scene.get('geometry', {}).pop('mobile_base', None)
            save(path, scene)
    for url in obsolete_urls - protected_urls:
        if url and url.startswith('/assets/'):
            retire(storage / url.removeprefix('/assets/'))
    for rid in removed:
        for folder in (storage / 'uploaded' / rid, storage / 'robots' / rid):
            retire(folder)
        for folder in (storage / 'manipulation4d').rglob('robot-' + rid):
            retire(folder)
        # Only named robot outputs inside explicitly configured runtime caches.
        # No shared observations, SMPL/SMPL-X, source hands, or scene data.
        for root in map(Path, config.get('robot_cache_roots', [])):
            if not root.exists():
                continue
            for folder in list(root.rglob(rid)):
                if folder.parent.name in {'operation-accuracy', 'initial-visuals'} or folder.parent.parent.name == 'mobile' or root.name in {'gripper-jobs', 'gripper-height'}:
                    retire(folder, root)
            for folder in list(root.rglob(rid + '-result')):
                if folder.parent.name == 'retarget':
                    retire(folder, root)
    write(trash / 'deletion.json', {'robot_ids': sorted(removed), 'metadata': backups, 'files': moved,
                                  'shared_human_priors_preserved': True})
    return {'deleted_robot_ids': sorted(removed), 'removed_cache_artifacts': len(moved),
            'smpl_preserved': True}
