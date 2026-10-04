"""Nymeria clock and four-endpoint contract; source samples remain unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation
from .xsens_clock import recording_clock,nearest_source_rows
from .orientation import source_semantic_frames
C=np.array([[0,1,0],[0,0,1],[1,0,0]],float)
ROLES=[14,10,21,17]

def read_source(path):
    with np.load(path,allow_pickle=False) as archive:d={k:archive[k].copy() for k in archive.files}
    if 'segment_tXYZ' in d:
        clock,report=recording_clock(d['timestamps_us'])
        query=np.arange(int(np.floor((clock[-1]-clock[0])*30/1e6))+1)*1e6/30+clock[0]
        rows=nearest_source_rows(clock,query)
        p=(d['segment_tXYZ'][rows].reshape(-1,23,3)@C.T*100).astype(np.float32)
        q=d['segment_qWXYZ'][rows].reshape(-1,23,4)
        rot=Rotation.from_quat(q[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(-1,23,3,3)
        rest=Rotation.from_quat(d['tpose_segment_qWXYZ'].reshape(23,4)[:,[1,2,3,0]]).as_matrix()
        relative=C@(rot@rest[None].swapaxes(-1,-2))@C.T
        semantic=source_semantic_frames(d['tpose_segment_tXYZ'],relative[:,ROLES].astype(np.float32))
        result=dict(position_yup_cm=p[:,ROLES],semantic_rotation_yup=semantic,foot_contacts=d['foot_contacts'][rows]>.5,timestamp_us=clock[rows],sample_timestamp_us=query,raw_frame_indices=rows,fps=np.array(30.),clock_report=report)
    else:
        if float(d.get('fps',0))!=30:raise ValueError('Sparse NPZ must explicitly declare fps=30')
        result={k:d[k] for k in ('position_yup_cm','semantic_rotation_yup','foot_contacts')}
        n=len(result['position_yup_cm']);result.update(fps=np.array(30.),sample_timestamp_us=np.arange(n)*1e6/30,clock_report={'policy':'caller-declared-uniform-30Hz'})
        if 'sample_timestamp_us' in d:result['sample_timestamp_us']=d['sample_timestamp_us']
        if not np.allclose(np.diff(result['sample_timestamp_us']),1e6/30,atol=1.,rtol=0):raise ValueError('Sparse input must have uniform 30Hz sample times')
    n=len(result['position_yup_cm'])
    if n<3 or result['position_yup_cm'].shape!=(n,4,3) or result['semantic_rotation_yup'].shape!=(n,4,3,3) or result['foot_contacts'].shape!=(n,4):raise ValueError('Invalid four-pose/contact shapes')
    for k in ('position_yup_cm','semantic_rotation_yup','sample_timestamp_us'):
        if not np.isfinite(result[k]).all():raise ValueError('Non-finite source '+k)
    rotations=result['semantic_rotation_yup']
    if not np.allclose(rotations.swapaxes(-1,-2)@rotations,np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(rotations),1.,atol=1e-5):raise ValueError('Semantic frames must be proper rotations')
    return result

def endpoint_indices(spec,robot):
    from greenwich_pipelines.robot_semantics import resolve_endpoints
    return resolve_endpoints(spec,robot.get('endpoints'))

def windows(source,window_frames=900):
    count=len(source['position_yup_cm'])
    for start in range(0,count,window_frames):
        end=min(start+window_frames,count)
        if end-start<3:
            yield start,end,None
        else:yield start,end,{k:(v[start:end] if isinstance(v,np.ndarray) and v.ndim and len(v)==count else v) for k,v in source.items()}
