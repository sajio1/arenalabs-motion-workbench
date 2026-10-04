"""A viewer-neutral result index; native numerical encodings are retained."""
from pathlib import Path
import hashlib,json

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(path)

def artifact(root,path,kind,**metadata):
    path=Path(path)
    return dict(path=path.relative_to(root).as_posix(),kind=kind,sha256=sha256(path),**metadata)

def inspect_native(path,pipeline):
    import numpy as np
    with np.load(path,allow_pickle=False) as d:
        keys={'fullbody':('q','root_t','rot6d'),'nymeria':('q','root_t','rot6d'),'gento':('qpos','actual_end_effector_positions')}[pipeline]
        for key in keys:
            if key not in d or not np.isfinite(d[key]).all():raise ValueError(f'Non-finite/missing saved {key}: {path}')
        frames=len(d[keys[0]])
        if frames<3:raise ValueError('At least three output frames required')
        if any(len(d[k])!=frames for k in keys):raise ValueError('Inconsistent output lengths')
        return frames
