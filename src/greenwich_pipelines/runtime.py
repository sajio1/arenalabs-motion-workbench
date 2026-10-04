"""Explicit local checkpoints; no hidden model download during a job."""
from pathlib import Path
import os
from .config import read_json,resolve

def configure(path):
    path=Path(path);d=read_json(path)
    if set(d)-{'weights','hf_home','cache','device'}:raise ValueError('Unknown runtime keys')
    weights=resolve(path.parent,d['weights'])
    for name in ('config.json','model.safetensors'):
        if not (weights/name).is_file():raise FileNotFoundError(weights/name)
    os.environ['GREENWICH_WEIGHTS']=str(weights)
    os.environ['ALPHAMOTION_DEVICE']=d.get('device','cuda')
    os.environ['ALPHAMOTION_CACHE']=str(resolve(path.parent,d.get('cache','../.cache')))
    if d.get('hf_home'):os.environ['HF_HOME']=str(resolve(path.parent,d['hf_home']))
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    return d,weights

def load_model():
    from alphamotion.engine.greenwich import Greenwich
    return Greenwich.load(model='prime',run_dir=Path(os.environ['GREENWICH_WEIGHTS']),device=os.environ['ALPHAMOTION_DEVICE'])
