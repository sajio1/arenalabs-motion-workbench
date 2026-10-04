"""Portable, explicit job contract. Relative paths belong to their JSON file."""
from dataclasses import dataclass
from pathlib import Path
import json
import re

PIPELINES=('fullbody','nymeria','gento')
ROOT=Path(__file__).resolve().parents[2]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def resolve(base,value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()

def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value):
        raise ValueError('IDs must be safe ASCII basenames')
    return value

@dataclass(frozen=True)
class Job:
    path: Path
    id: str
    pipeline: str
    source: Path
    robots: Path
    output: Path
    runtime: Path
    acceptance: str|None=None
    window_frames: int=900
    max_seconds: float|None=None

    @classmethod
    def load(cls,path):
        path=Path(path).resolve();d=read_json(path)
        allowed={'schema','id','pipeline','source','robots','output','runtime','fps','acceptance','window_frames','max_seconds'}
        if set(d)-allowed:raise ValueError(f'Unknown job keys: {sorted(set(d)-allowed)}')
        if d.get('schema')!='greenwich.job.v1':raise ValueError('Expected greenwich.job.v1')
        if d['pipeline'] not in PIPELINES:raise ValueError('Unknown pipeline')
        if d.get('fps',30)!=30:raise ValueError('This integration supports 30 Hz only')
        window=d.get('window_frames',900)
        if not isinstance(window,int) or isinstance(window,bool) or window<3:raise ValueError('window_frames >=3 required')
        duration=d.get('max_seconds')
        if duration is not None and (not isinstance(duration,(int,float)) or duration<=0):raise ValueError('Invalid max_seconds')
        if duration is not None and d['pipeline']!='fullbody':raise ValueError('max_seconds is supported only for fullbody')
        if 'window_frames' in d and d['pipeline']!='nymeria':raise ValueError('window_frames is supported only for nymeria')
        acceptance=d.get('acceptance')
        if acceptance is not None and (not isinstance(acceptance,str) or ':' not in acceptance):raise ValueError('Acceptance uses module:function')
        paths={k:resolve(path.parent,d[k]) for k in ('source','robots','output','runtime')}
        if paths['output'] in (paths['source'],paths['robots'],paths['runtime']):raise ValueError('Output must not overwrite input')
        return cls(path,safe_id(d['id']),d['pipeline'],**paths,acceptance=acceptance,window_frames=window,max_seconds=duration)

def robot_rows(path):
    path=Path(path);rows=read_json(path)
    if not isinstance(rows,list) or not rows:raise ValueError('Robot roster must be a nonempty list')
    seen=set();out=[]
    forbidden={'weights','smoothing','seed','root_height','solver','policy','iterations','loss'}
    for row in rows:
        if forbidden & set(row):raise ValueError('Robot roster may contain physical metadata only')
        row=dict(row);key=safe_id(row.get('body',row.get('name')))
        if key in seen:raise ValueError('Duplicate robot body')
        seen.add(key);row['body']=key;row.setdefault('name',key)
        for k in ('model','xml','source','metadata'):
            if row.get(k):row[k]=str(resolve(path.parent,row[k]))
        if not any(row.get(k) for k in ('model','xml','metadata')):raise ValueError('Explicit robot model or metadata required')
        out.append(row)
    return out
