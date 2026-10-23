"""Enumerate canonical RGB-D files for lossless, independently downloadable bundles."""
import json
import os
from pathlib import Path, PurePosixPath


def bundle_files(folder):
    folder = Path(folder).resolve()

    def within(relative):
        if not isinstance(relative, str) or '\\' in relative or ':' in relative:
            raise ValueError('Invalid geometry bundle path')
        parts = PurePosixPath(relative)
        if parts.is_absolute() or '..' in parts.parts or not parts.parts:
            raise ValueError('Invalid geometry bundle path')
        candidate = folder / relative
        if not candidate.resolve().is_relative_to(folder) or not candidate.exists():
            raise ValueError('Geometry bundle dependency missing or outside its root')
        return candidate

    index = within('index.json')
    meta = json.loads(index.read_text())
    if meta.get('schema') != 'rgbd-geometry-v1':
        raise ValueError('Unsupported canonical geometry bundle')
    selected = [index, *(within(meta[key]) for key in ('rgb', 'cameras', 'masks'))]
    for key in ('depth', 'source_human_layer'):
        if meta.get(key):
            dependency = within(meta[key])
            relative = dependency.relative_to(folder)
            if not relative.parts[0].endswith('.zarr'):
                raise ValueError('Geometry depth must use a canonical Zarr group')
            selected.append(within(relative.parts[0]))
    files = {}
    for source in selected:
        if source.is_file():
            files[source.relative_to(folder).as_posix()] = source
            continue
        for base, dirs, names in os.walk(source, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != '.artifact-revisions')
            if any((Path(base)/name).is_symlink() for name in dirs):
                raise ValueError('Geometry bundle cannot contain linked directories')
            for name in sorted(names):
                path = Path(base) / name
                if not path.is_file() or not path.resolve().is_relative_to(folder):
                    raise ValueError('Geometry bundle contains an invalid file')
                files[path.relative_to(folder).as_posix()] = path
    return [(path, relative) for relative, path in sorted(files.items())]
