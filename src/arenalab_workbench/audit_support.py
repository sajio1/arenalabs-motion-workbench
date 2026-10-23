"""Independent support-area diagnostics from native FK and vendor sole geometry."""
import json,sys
from pathlib import Path
import numpy as np,torch
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from greenwich_pipelines.pipelines.nymeria.embodiment_exact import load_robot
from greenwich_pipelines.pipelines.nymeria.physical_fk import PhysicalFK
from greenwich_pipelines.pipelines.nymeria.sole_geometry import sole_surface
from arenalab_workbench.support_reference import support_reference
from arenalab_workbench.human_reference import load_reference, support_model

def audit(work, robot, path):
    row_path=work/'retarget'/f'{robot}-robot.json'
    if not row_path.exists():row_path=work/f'{robot}-robot.json'
    row=json.loads(row_path.read_text())[0]
    body=load_robot(row); fk=PhysicalFK(body,'cpu',torch.float64)
    d=np.load(path); ids=[body.spec.joint_names.index(row['endpoints'][s+'_ankle']) for s in ['left','right']]
    soles,hulls,valid=sole_surface(body,ids)
    q=torch.tensor(d['q'],dtype=torch.float64).flatten(1)[:,fk.active_indices]
    root_R=rot6d_to_matrix(torch.tensor(d['rot6d'],dtype=torch.float64))[:,fk.root]
    with torch.no_grad():R,p=fk(q,root_R)
    R=R.numpy();world=p.numpy()+d['root_t'][:,None]/100
    error=float(np.max(np.abs(world-d['world_position_cm']/100)))
    if error>1e-6:raise ValueError('Independent FK differs from exported motion')
    floor=float(d['ground_height_m']) if 'ground_height_m' in d else 0.
    _,conf,_,_=support_reference(load_reference(work),support_model(work))
    corners=(np.einsum('tfij,fkj->tfki',R[:,ids],soles)+world[:,ids,None])[...,1]-floor
    envelope=(np.einsum('tfij,fkj->tfki',R[:,ids],hulls)+world[:,ids,None])[...,1]-floor
    mask=conf>.7; span=np.ptp(corners,axis=-1)
    def stats(x):return dict(mean=float(np.mean(x)),p95=float(np.percentile(x,95)),max=float(np.max(x)))
    wrist_ids=[body.spec.joint_names.index(row['endpoints'][s+'_wrist']) for s in ['left','right']]
    actual=world[:,wrist_ids]+np.einsum('tsij,tsj->tsi',R[:,wrist_ids],d['operation_local_yup_m'])
    return dict(ground_height_m=floor,independent_fk_max_error_m=error,
        support_samples=int(mask.sum()),support_corner_height_mm=stats(np.abs(corners[mask])*1000),
        support_heel_toe_height_difference_mm=stats(span[mask]*1000),
        support_frames_with_corner_above_20mm=int(np.sum((corners.max(-1)>.02)&mask)),
        minimum_envelope_height_mm=float(envelope[:,valid].min()*1000),
        palm_error_mm=stats(np.linalg.norm(actual-d['operation_target_yup_m'],axis=-1)*1000),
        knee_range_deg=np.rad2deg([d['knee_flexion_rad'].min(0),d['knee_flexion_rad'].max(0)]).tolist())

if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])),indent=2))
