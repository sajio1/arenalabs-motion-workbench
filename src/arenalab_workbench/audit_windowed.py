"""Report seam metrics without changing any generated or source motion."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from arenalab_workbench.window_plan import digest,save_json

def stats(x):
    x=np.asarray(x)
    return dict(mean=float(x.mean()),p95=float(np.percentile(x,95)),max=float(x.max())) if x.size else None

def audit(root,output,comparison=None):
    root=Path(root);output=Path(output)
    data=np.load(output/'egoallo_smplh.npz');source=np.load(root/'observations.npz')
    manifest=json.loads((output/'manifest.json').read_text());seams=json.loads((output/'seams.json').read_text())
    n=len(data['root_pose']);assert n==len(source['timestamps_s'])
    assert digest(root/'observations.npz')==manifest['source_sha256']
    assert all(np.isfinite(data[k]).all() for k in data.files)
    positions=np.concatenate([data['root_pose'][:,None,4:],data['joint_poses'][...,4:]],axis=1)
    velocity=np.linalg.norm(np.diff(positions,axis=0),axis=-1)*30
    acceleration=np.linalg.norm(np.diff(positions,n=2,axis=0),axis=-1)*900
    quats=data['local_quats'];q=Rotation.from_quat(quats.reshape(-1,4)[:,[1,2,3,0]])
    qmat=q.as_matrix().reshape(n,51,3,3)
    angular=np.linalg.norm(Rotation.from_matrix((np.swapaxes(qmat[:-1],-1,-2)@qmat[1:]).reshape(-1,3,3)).as_rotvec().reshape(n-1,51,3),axis=-1)*30
    feet=positions[:,[7,8,10,11]]
    contact=data['contacts'][:,[6,7,9,10]]
    slip=np.linalg.norm(np.diff(feet[...,:2],axis=0),axis=-1)*30
    mask=np.minimum(contact[:-1],contact[1:])>.8
    wrist=np.linalg.norm(data['joint_poses'][:,[19,20],4:]-source['wrist_position'],axis=-1)*1000
    report=dict(frames=n,source_unchanged=True,finite=True,shape_std_max=float(data['betas'].std(0).max()),
        source_wrist_prior_error_mm=stats(wrist),joint_velocity_m_s=stats(velocity),joint_acceleration_m_s2=stats(acceleration),
        contact_foot_horizontal_speed_m_s=stats(slip[mask]),joint_angular_velocity_rad_s=stats(angular),seams=[])
    for seam in seams:
        for frame in [seam['start'],seam['end']]:
            lo=max(0,frame-5);hi=min(n-2,frame+5)
            report['seams'].append(dict(frame=frame,pelvis_speed_m_s=stats(velocity[lo:hi,0]),
                joint_speed_m_s=stats(velocity[lo:hi]),joint_acceleration_m_s2=stats(acceleration[lo:hi]),
                joint_angular_velocity_rad_s=stats(angular[lo:hi]),wrist_prior_error_mm=stats(wrist[lo:hi])))
    if comparison:
        other=np.load(Path(comparison)/'egoallo_smplh.npz')
        report['different_cut_joint_position_mm']=stats(np.linalg.norm(data['joint_poses'][...,4:]-other['joint_poses'][...,4:],axis=-1)*1000)
        report['different_cut_pelvis_mm']=stats(np.linalg.norm(data['root_pose'][:,4:]-other['root_pose'][:,4:],axis=-1)*1000)
    save_json(output/'audit.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':audit(*sys.argv[1:])
