"""Optical poses in unchanged Xsens source world, shared by every target robot."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

def prepare_camera(row,work,root):
    import sys
    sys.path.insert(0,'/opt/arenalab/src')
    from greenwich_pipelines.pipelines.nymeria import xsens_clock
    sequence=row.get('metadata',{}).get('Sequence',Path(row['source']).parent.name)
    profile=Path(root)/'nymeria-calibration'/(sequence+'-alignment.npz')
    if not profile.exists():raise ValueError('Nymeria head/device alignment is missing')
    fixed=np.load(profile)['T_head_camera'];n=int(row['frames']);start=int(row.get('source_window',{}).get('start_frame',0))
    with np.load(row['source']) as data:
        times,_=xsens_clock.recording_clock(data['timestamps_us'])
        query=times[0]+np.arange(start,start+n)*1e6/30
        if query[-1]>times[-1]:raise ValueError('Camera window exceeds source recording')
        idx=xsens_clock.nearest_source_rows(times,query)
        p=data['segment_tXYZ'][idx].reshape(n,23,3)[:,6]
        q=data['segment_qWXYZ'][idx].reshape(n,23,4)[:,6]
    head=np.repeat(np.eye(4)[None],n,0);head[:,:3,3]=p;head[:,:3,:3]=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    poses=head@fixed
    K=np.load(profile.with_name(sequence+'.npz'))['intrinsic']
    np.savez_compressed(Path(work)/'cameras.npz',poses=poses,intrinsic=K,source_rows=idx)
    report=json.loads(profile.with_suffix('.json').read_text())
    report.update(world='Unmodified Xsens source world',camera_equation='T_X_C(t) = T_X_H(t) @ inverse(T_D_H) @ T_D_Cupright',source_rows=idx.tolist(),roundtrip_max_error=float(np.max(abs(poses@np.linalg.inv(fixed)-head))))
    (Path(work)/'camera-calibration.json').write_text(json.dumps(report,indent=2))
