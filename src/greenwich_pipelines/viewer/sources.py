"""Prepare explicit source windows without modifying the original recording."""
from pathlib import Path
import numpy as np

def prepare_source(clip,folder):
    path=Path(clip['source'])
    window=clip.get('source_window')
    if not window:return path
    if clip['page']!='localmanipulation':raise ValueError('Window preparation is defined only for Nymeria')
    from greenwich_pipelines.pipelines.nymeria.input import read_source
    source=read_source(path);start=window['start_frame'];end=window['end_frame']
    if not 0<=start<end<=len(source['position_yup_cm']):raise ValueError('Source window outside recording')
    count=len(source['position_yup_cm'])
    segment={k:(v[start:end] if isinstance(v,np.ndarray) and v.ndim and len(v)==count else v) for k,v in source.items() if k!='clock_report'}
    target=Path(folder)/'source-window.npz';np.savez_compressed(target,**segment)
    return target
