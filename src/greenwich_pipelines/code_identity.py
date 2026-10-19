"""Fingerprint conversion dependencies without coupling caches to the web/API code.

Follow repository imports and literal subprocess module names without importing
GPU libraries. Include policy files, plus package initializers and re-exports.
"""
import ast
import hashlib
from importlib.util import resolve_name
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]


def dependency_files(modules, source=SOURCE):
    source = Path(source)
    seen = set()
    files = set()
    pending = list(modules)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        relative = Path(*name.split('.'))
        path = source / relative.with_suffix('.py')
        package = False
        if not path.is_file():
            path = source / relative / '__init__.py'
            package = True
        if not path.is_file():
            continue  # External dependencies are pinned by the runtime deployment.
        files.add(path)
        components = name.split('.')
        pending.extend('.'.join(components[:i]) for i in range(1, len(components)))
        context = name if package else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = resolve_name('.' * node.level + (node.module or ''), context) if node.level else node.module
                if target:
                    pending.append(target)
                    pending.extend(target + '.' + alias.name for alias in node.names if alias.name != '*')
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # E.g. python -m ...backend; worker selects its runner dynamically.
                if node.value.startswith(('greenwich_', 'alphamotion.')) and all(p.isidentifier() for p in node.value.split('.')):
                    pending.append(node.value)
    return sorted(files)


def conversion_code_signature(pipeline, source=SOURCE):
    roots = ['greenwich_pipelines.worker',
             'greenwich_pipelines.pipelines.' + pipeline + '.runner',
             'greenwich_pipelines.viewer.sources', 'greenwich_pipelines.viewer.export']
    source = Path(source)
    files = dependency_files(roots, source)
    policies = source.parent / 'configs' / 'policies'
    files += sorted(policies.glob('*.json'))
    digest = hashlib.sha256(b'conversion-dependency-v1')
    digest.update(Path(__file__).read_bytes())
    for path in files:
        digest.update(path.relative_to(source.parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
