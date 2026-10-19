"""Production code must run both installed and from an immutable job snapshot."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from arenalab_workbench.settings import Paths
import arenalab_workbench


def test_deployment_overrides_do_not_create_runtime(tmp_path):
    config = Paths({'ARENA_WORKBENCH_ROOT': str(tmp_path / 'compute'),
                    'ARENA_STORE': str(tmp_path / 'assets'),
                    'ARENA_CONFIG_DIR': str(tmp_path / 'config'),
                    'ARENA_SAM_WEIGHTS': str(tmp_path / 'weights.pt')})
    assert config.scene_python == tmp_path / 'compute/venv/bin/python'
    assert config.sam_weights == tmp_path / 'weights.pt'
    assert config.viewer_config == tmp_path / 'config/viewer.cloud.json'
    assert not list(tmp_path.iterdir())


def test_frozen_package_imports_its_own_helpers(tmp_path):
    source = Path(arenalab_workbench.__file__).parent
    snapshot = tmp_path / 'code'
    shutil.copytree(source, snapshot / 'arenalab_workbench', ignore=shutil.ignore_patterns('__pycache__'))
    env = dict(os.environ, PYTHONPATH=str(snapshot))
    code = "import json; import arenalab_workbench.discovery_output as m; print(json.dumps({'file':m.__file__,'objects':m.normalize_discovery({'objects':['cup']})['objects']}))"
    result = subprocess.check_output([sys.executable, '-c', code], cwd=tmp_path, env=env, text=True)
    output = json.loads(result)
    assert Path(output['file']).is_relative_to(snapshot)
    assert output['objects'] == ['cup']
