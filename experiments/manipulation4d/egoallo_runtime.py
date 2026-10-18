"""Isolated CUDA JAX runtime. Does not change inference or source policy."""
from pathlib import Path
import os

RUNTIME = Path('/home/ubuntu/arenalab-experiments/egoallo/runtime-jax061')

def gpu_environment(base=None):
    env = dict(os.environ if base is None else base)
    deps = RUNTIME / 'deps'
    if not (deps / 'jax_plugins').is_dir():
        raise RuntimeError(f'Missing isolated JAX CUDA dependencies: {deps}')
    cache = RUNTIME / 'jax-cache'
    cache.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','3'), JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_DEFAULT_MATMUL_PRECISION='highest',
        JAX_COMPILATION_CACHE_DIR=str(cache))
    for key, prefix in [('PYTHONPATH', str(deps)),
                        ('LD_LIBRARY_PATH', str(deps / 'nvidia/cudnn/lib')),
                        ('PATH', str(deps / 'nvidia/cuda_nvcc/bin'))]:
        env[key] = prefix + (os.pathsep + env[key] if env.get(key) else '')
    return env
