"""Sparse Nymeria adapter. Body/feet are reconstructed, fingers are not inferred."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from greenwich_pipelines.pipelines.nymeria import xsens_clock


def prepare(public, row, work, save):
    work=Path(work); work.mkdir(parents=True,exist_ok=True)
    n=int(row['frames']); start=int(row.get('source_window',{}).get('start_frame',0))
    with np.load(row['source']) as data:
        times, clock=xsens_clock.recording_clock(data['timestamps_us'])
        query=times[0]+np.arange(start,start+n)*1e6/30
        if query[-1]>times[-1]: raise ValueError('Nymeria window exceeds source recording')
        indices=xsens_clock.nearest_source_rows(times,query)
        positions=data['segment_tXYZ'][indices].reshape(n,23,3)
        rotation=Rotation.from_quat(data['segment_qWXYZ'][indices].reshape(-1,4)[:,[1,2,3,0]]).as_matrix().reshape(n,23,3,3)
        rest=Rotation.from_quat(data['tpose_segment_qWXYZ'].reshape(23,4)[:,[1,2,3,0]]).as_matrix()
        tpose=data['tpose_segment_tXYZ'].reshape(23,3)
    relative=rotation@rest.transpose(0,2,1)
    basis=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    head_rotation=relative[:,6]@basis
    forward=[]; normal=[]
    for wrist,elbow in [(14,13),(10,9)]:
        direction=tpose[wrist]-tpose[elbow]; direction/=np.linalg.norm(direction)
        forward.append(np.einsum('tij,j->ti',relative[:,wrist],direction))
        normal.append(np.einsum('tij,j->ti',relative[:,wrist],[0,0,-1.]))
    wrists=positions[:,[14,10]]
    np.savez_compressed(work/'observations.npz',head_position=np.r_[positions[:1,6],positions[:,6]],
        head_rotation=np.concatenate([head_rotation[:1],head_rotation]),wrist_position=wrists,
        palm_position=wrists,palm_forward=np.stack(forward,1),palm_normal=np.stack(normal,1),
        timestamps_s=np.arange(n)/30,floor_z=np.array(0.))
    (work/'operation-accuracy').mkdir(exist_ok=True)
    np.savez_compressed(work/'operation-accuracy/source-targets.npz',points=wrists[:,:,None])
    # Keep observed elbows for auditing only; never feed recorded pelvis or feet.
    np.savez_compressed(work/'observed_elbows.npz',positions=positions[:,[13,9]],source_rows=indices)
    save(work/'input_manifest.json',dict(dataset='Nymeria',clip=public['id'],source=row['source'],
        source_window=row.get('source_window'),frames=n,fps=30,video=row.get('video'),
        source_sha256=hashlib.sha256(Path(row['source']).read_bytes()).hexdigest(),
        source_trajectory_edited=False,body_inputs=['head pose','left/right wrist pose'],
        fingers='Unavailable; no dexterous hand attached',end_effector='none',
        cpf_policy='Anatomical head proxy; no invented camera extrinsic',
        floor_policy='Xsens source z=0; reconstructed support estimate kept separate',clock=clock))
    return work
