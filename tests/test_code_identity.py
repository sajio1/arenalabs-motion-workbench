from pathlib import Path
from greenwich_pipelines.code_identity import dependency_files, conversion_code_signature


def test_conversion_dependencies_ignore_ui_and_follow_relative_workers(tmp_path):
    src = tmp_path / 'src'
    package = src / 'greenwich_pipelines'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'worker.py').write_text('from .math import solve\n')
    (package / 'math.py').write_text('solve = 1\n')
    ui = package / 'viewer'
    ui.mkdir()
    (ui / 'server.py').write_text('button = "old"\n')
    files = dependency_files(['greenwich_pipelines.worker'], src)
    assert package / 'math.py' in files
    assert ui / 'server.py' not in files
    before = conversion_code_signature('fullbody', src)
    (ui / 'server.py').write_text('button = "new"\n')
    assert conversion_code_signature('fullbody', src) == before
    (package / 'math.py').write_text('solve = 2\n')
    assert conversion_code_signature('fullbody', src) != before
