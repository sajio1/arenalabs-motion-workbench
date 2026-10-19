"""Deployment paths. Importing this module never creates directories or starts jobs.

Override ARENA_WORKBENCH_ROOT, ARENA_STORE, ARENA_CONFIG_DIR and interpreter
variables in the service environment. Cloud defaults are centralized here only.
"""
from pathlib import Path
import os

class Paths:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.workbench = Path(env.get('ARENA_WORKBENCH_ROOT', '/home/ubuntu/arenalab-experiments/manipulation4d'))
        self.store = Path(env.get('ARENA_STORE', '/opt/arenalab/store'))
        self.config = Path(env.get('ARENA_CONFIG_DIR', '/opt/arenalab/configs'))
        self.viewer_config = Path(env.get('ARENA_VIEWER_CONFIG', str(self.config / 'viewer.cloud.json')))
        self.source = Path(env.get('ARENA_SOURCE_ROOT', '/opt/arenalab/src'))
        self.egoallo = Path(env.get('ARENA_EGOALLO_ROOT', str(self.workbench.parent / 'egoallo')))
        self.da3 = Path(env.get('ARENA_DA3_ROOT', str(self.workbench.parent / 'da3')))
        self.scene_python = Path(env.get('ARENA_SCENE_PYTHON', str(self.workbench / 'venv/bin/python')))
        self.motion_python = Path(env.get('ARENA_MOTION_PYTHON', '/opt/arenalab/.venv/bin/python'))
        self.egoallo_python = Path(env.get('ARENA_EGOALLO_PYTHON', str(self.egoallo / 'venv/bin/python')))
        self.da3_python = Path(env.get('ARENA_DA3_PYTHON', str(self.da3 / 'venv/bin/python')))
        self.jax = Path(env.get('ARENA_JAX_ROOT', str(self.egoallo / 'runtime-jax061')))
        self.sam = Path(env.get('ARENA_SAM_ROOT', str(self.workbench / 'sam3')))
        self.sam_weights = Path(env.get('ARENA_SAM_WEIGHTS', str(self.workbench / 'sam31-weights/sam3.1_multiplex.pt')))
        self.umi_calibration = Path(env.get('ARENA_UMI_CALIBRATION', '/home/ubuntu/wuji-ego-pilot/estimated-scene.json'))

    def pythonpath(self, *extra):
        # The package parent also supports immutable per-job worker snapshots.
        parts = [str(p) for p in extra] + [str(Path(__file__).resolve().parent.parent), str(self.source)]
        parts += os.environ.get('PYTHONPATH', '').split(os.pathsep)
        return os.pathsep.join(dict.fromkeys(p for p in parts if p))

paths = Paths()
