"""Explicitly package the configured demo and its local dependencies for Linux.

Does not launch instances, upload files, or include SSH/API credentials.
"""
import argparse
import io
import json
import re
import tarfile
from pathlib import Path


def package(config_path, output, remote_root='/opt/arenalab'):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding='utf-8'))
    store = Path(config['storage']).resolve()
    files = {}
    generated = {}
    processing = set()

    def destination(path):
        path = path.resolve()
        if path.is_relative_to(store):
            return 'store/' + path.relative_to(store).as_posix()
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
        # Mirror outside-repo paths to preserve relative mesh/include references.
        return 'private/resources/' + path.as_posix().replace(':', '')

    def rewrite(value):
        if isinstance(value, dict):
            return {k: ([] if k == 'library_roots' else rewrite(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if isinstance(value, str) and re.match(r'^[A-Za-z]:[/\\]', value):
            p = Path(value)
            if p.exists():
                if p.is_file():
                    add(p)
                return remote_root + '/' + destination(p)
        return value

    def add(path):
        path = path.resolve()
        if path in processing:
            return
        processing.add(path)
        if path.is_dir():
            for child in path.rglob('*'):
                if child.is_file() and not set(child.parts) & {'__pycache__', '.git', 'node_modules', '.pytest_cache'}:
                    add(child)
            return
        name = destination(path)
        if path.suffix.lower() == '.xml':
            content = path.read_text(encoding='utf-8')
            content = re.sub(r'(?<=\")[A-Za-z]:[/\\][^\"]+(?=\")', lambda m: rewrite(m.group()), content)
            generated[name] = content.encode()
            return
        if path.suffix.lower() == '.json':
            try:
                generated[name] = json.dumps(rewrite(json.loads(path.read_text(encoding='utf-8'))), ensure_ascii=False).encode()
                return
            except (ValueError, UnicodeError):
                pass
        files[name] = path

    # Capture only code/build/runtime assets, never unrelated working directories.
    for name in ('src', 'assets', 'configs', 'third_party', 'web/dist'):
        add(root / name)
    for name in ('pyproject.toml', 'NOTICE.md', 'README.md'):
        add(root / name)
    add(root / 'tools/bootstrap_cloud_demo.sh')
    add(store)
    runtime = json.loads(Path(config['runtime']).read_text(encoding='utf-8'))
    for key in ('weights', 'hf_home'):
        add(Path(runtime[key]))
    for robot in config['robots']:
        add(Path(robot['pipeline_row']['xml']).parent)
    # Uploaded registry can also reference models outside its own directory.
    registry_path = store / 'uploaded/registry.json'
    if registry_path.exists():
        for robot in json.loads(registry_path.read_text()):
            if robot.get('pipeline_row', {}).get('xml'):
                add(Path(robot['pipeline_row']['xml']).parent)
    for clip in config['clips']:
        add(Path(clip['source']))
        if clip.get('video'):
            add(Path(clip['video']['path']))
    cloud = rewrite(config)
    cloud['storage'] = remote_root + '/store'
    cloud['runtime'] = remote_root + '/configs/runtime.cloud.json'
    cloud['library_roots'] = []
    # Preserve verified existing playback; paths/signatures necessarily change.
    old = json.loads((store / 'viewer-conversions.json').read_text())
    from greenwich_pipelines.viewer.server import conversion_signature
    robots = {r['id']: r for r in config['robots']}
    if registry_path.exists():
        robots.update({r['id']: r for r in json.loads(registry_path.read_text())})
    clips = {c['id']: c for c in config['clips']}
    verified = []
    for entry in old.values():
        clip, robot = clips.get(entry['clip_id']), robots.get(entry['robot_id'])
        if clip and robot and entry['signature'] == conversion_signature(clip, robot):
            playback = store / entry['url'].removeprefix('/assets/')
            if playback.is_file():
                cloud_clip = next(c for c in cloud['clips'] if c['id'] == clip['id'])
                cloud_clip.setdefault('cache', {})[robot['id']] = entry['url']
                verified.append(entry['url'])
    runtime = rewrite(runtime)
    runtime['cache'] = remote_root + '/private/runtime-cache'
    for name, obj in [('configs/viewer.cloud.json', cloud), ('configs/runtime.cloud.json', runtime), ('store/viewer-conversions.json', {})]:
        generated[name] = json.dumps(obj, ensure_ascii=False).encode()
    # HuggingFace Windows caches may use symlinks: archive dereferences them.
    manifest = {'files': len(files) + len(generated), 'bytes': sum(p.stat().st_size for p in files.values()) + sum(map(len, generated.values())), 'verified_playbacks': verified, 'clips': len(cloud['clips'])}
    generated['deployment-manifest.json'] = json.dumps(manifest, indent=2).encode()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(manifest), flush=True)
    with tarfile.open(output, 'w', dereference=True) as archive:
        for name, path in sorted(files.items()):
            if name not in generated:
                archive.add(path, arcname=name, recursive=False)
        for name, content in sorted(generated.items()):
            info = tarfile.TarInfo(name); info.size = len(content); info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Packaged {output.stat().st_size / 2**30:.2f} GiB: {output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('configs/viewer.local.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    package(a.config, a.output)
