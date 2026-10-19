"""Exact-code reuse of independent AlphaMotion frames produced by body windows."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import torch


def target_key(codec,spec,dof):
    h=hashlib.sha256(json.dumps(dict(model=codec.name,config=codec.cfg,names=spec.joint_names),sort_keys=True).encode())
    weights=Path(os.environ['GREENWICH_WEIGHTS'])/'model.safetensors';identity=hashlib.sha256()
    with weights.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):identity.update(block)
    h.update(identity.digest())
    for a in [spec.parents,spec.rest_offsets,dof]:h.update(np.asarray(a).tobytes())
    return h.hexdigest()


@torch.no_grad()
def decode_cached(codec,codes,spec,dof,frame_ids,timestamps,cache_dir,chunk=32):
    n=len(codes);files=sorted(Path(cache_dir).glob('chunk-*/robot.npz')) if cache_dir else []
    if not files:
        out=[codec.decode_full(codes[k:k+chunk],spec,dof,contact=True) for k in range(0,n,chunk)]
        return (*(torch.cat([a[i] for a in out]) for i in range(3)),dict(reused_frames=0,decoded_frames=n,frames=n))
    key=target_key(codec,spec,dof);saved={};ids={int(f):i for i,f in enumerate(frame_ids)}
    cpu=codes.cpu().numpy()
    if cache_dir:
        for p in files:
            with np.load(p) as d:
                if str(d['target_key'])!=key:continue
                # NpzFile is lazy: indexing d[key] repeatedly inflates the whole
                # compressed member each time, even when selecting one frame.
                cached={k:d[k] for k in ['frame_ids','timestamps_s','codes',
                                         'raw_rot6d','decoder_position','model_contact']}
                for j,f in enumerate(cached['frame_ids']):
                    i=ids.get(int(f))
                    if i is not None and abs(float(cached['timestamps_s'][j])-float(timestamps[i]))<1e-7 and np.array_equal(cached['codes'][j],cpu[i]):
                        saved[i]=tuple(cached[k][j].copy() for k in ['raw_rot6d','decoder_position','model_contact'])
    missing=[i for i in range(n) if i not in saved];parts=[]
    for start in range(0,len(missing),chunk):
        indices=missing[start:start+chunk]
        out=codec.decode_full(codes[indices],spec,dof,contact=True)
        parts.append(out)
    example=tuple(v[0] for v in parts[0]) if parts else tuple(torch.as_tensor(a,device=codes.device) for a in saved[0])
    result=[torch.empty((n,*v.shape),dtype=v.dtype,device=codes.device) for v in example]
    if saved:
        indices=list(saved)
        dest=torch.as_tensor(indices,device=codes.device)
        for k,out in enumerate(result):
            values=np.stack([saved[i][k] for i in indices])
            out.index_copy_(0,dest,torch.as_tensor(values,device=out.device))
    if parts:
        dest=torch.as_tensor(missing,device=codes.device)
        for k,out in enumerate(result):
            out.index_copy_(0,dest,torch.cat([p[k] for p in parts]))
    return (*result,dict(reused_frames=len(saved),decoded_frames=len(missing),frames=n,criterion='Exact integer codes, target descriptor, model config, frame id and timestamp'))
