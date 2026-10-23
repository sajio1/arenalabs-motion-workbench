"""Explicit local model/cache configuration; never resolve or download weights."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
import time

@dataclass(frozen=True)
class Runtime:
    weights: Path
    cache: Path
    hf_home: Path | None = None
    device: str = 'cuda:0'

    @classmethod
    def read(cls, value=None):
        value = value or os.environ.get('ARENA_MOTION_RUNTIME') or os.environ.get('ARENA_GPU_MOTION_RUNTIME')
        if value is None:
            raise ValueError('Set request.runtime_config or ARENA_MOTION_RUNTIME to an explicit local runtime JSON')
        if isinstance(value, (str, Path)):
            path=Path(value).expanduser().resolve();data=json.loads(path.read_text());base=path.parent
        else:
            data=dict(value);base=Path.cwd()
        unknown=set(data)-{'weights','cache','hf_home','device'}
        if unknown:raise ValueError('Unknown motion runtime keys: '+str(sorted(unknown)))
        def local(key):
            p=Path(data[key]).expanduser()
            return (base/p).resolve() if not p.is_absolute() else p.resolve()
        if 'weights' not in data or 'cache' not in data:
            raise ValueError('Motion runtime requires explicit weights and cache directories')
        runtime=cls(local('weights'),local('cache'),local('hf_home') if data.get('hf_home') else None,('cuda:0' if data.get('device','cuda:0')=='cuda' else data.get('device','cuda:0')))
        for name in ('config.json','model.safetensors'):
            if not (runtime.weights/name).is_file():raise FileNotFoundError(runtime.weights/name)
        return runtime

    def configure(self):
        os.environ.update(GREENWICH_WEIGHTS=str(self.weights),ALPHAMOTION_DEVICE=self.device,
                          ALPHAMOTION_CACHE=str(self.cache),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        self.cache.mkdir(parents=True,exist_ok=True)
        if self.hf_home is not None:os.environ['HF_HOME']=str(self.hf_home)

    def identity(self):
        from .prior import sha
        return {name:sha(self.weights/name) for name in ('config.json','model.safetensors')}

    def load_codec(self):
        self.configure()
        import torch
        from alphamotion.engine.greenwich import Greenwich
        torch.set_num_threads(2)
        if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)
        start=time.perf_counter()
        codec=Greenwich.load(run_dir=self.weights,model='prime',device=self.device)
        if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)
        codec._grid_cold_load_s=time.perf_counter()-start
        codec._motion_model_identity=self.identity()
        return codec
