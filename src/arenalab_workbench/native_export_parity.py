"""Independent CPU check of Grid playback against saved native world joint states."""
from pathlib import Path
import json

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

DISPLAY_TO_WORLD=np.array([[1.,0,0],[0,0,-1],[0,1,0]])


def audit_native_export(row,qpos_path,playback_path,visual_path,operation_local):
    m=mj.MjModel.from_xml_path(row['xml']);d=mj.MjData(m)
    with np.load(qpos_path) as data:q=np.asarray(data['qpos'])
    with np.load(visual_path) as visual:ids=np.asarray(visual['native_body_geom_ids'],int)
    playback=json.loads(Path(playback_path).read_text());n=len(q);k=len(ids)
    if playback['frames']!=n or playback['fps']!=30:raise ValueError('Export clock mismatch')
    p=np.asarray(playback['positions'])[:,:k]@DISPLAY_TO_WORLD.T
    quat=np.asarray(playback['quaternions'])[:,:k]
    rot=DISPLAY_TO_WORLD@Rotation.from_quat(quat.reshape(-1,4)).as_matrix().reshape(n,k,3,3)
    if p.shape!=(n,k,3):raise ValueError('Export geometry count mismatch')
    task=np.asarray(playback['tracking']['actual'])@DISPLAY_TO_WORLD.T
    wrists=[m.body(row['endpoints'][s+'_wrist']).id for s in ('left','right')]
    locals_=np.asarray(operation_local)[:,:,0]
    joint_names=playback['joints']['names'];joint_values=np.asarray(playback['joints']['values'])
    if len(set(joint_names))!=len(joint_names):raise ValueError('Duplicate exported native joint name')
    by_name=dict(zip(joint_names,joint_values.T));joint_error=0.;checked_joints=0
    for j in range(m.njnt):
        if m.jnt_type[j] not in (mj.mjtJoint.mjJNT_HINGE,mj.mjtJoint.mjJNT_SLIDE):continue
        name=m.joint(j).name
        if name not in by_name:raise ValueError('Missing native scalar joint in Grid output: '+name)
        joint_error=max(joint_error,float(np.max(abs(by_name[name]-q[:,m.jnt_qposadr[j]]))));checked_joints+=1
    p_error=r_error=task_error=0.
    for t,pose in enumerate(q):
        d.qpos[:]=pose;mj.mj_forward(m,d)
        p_error=max(p_error,float(np.max(np.linalg.norm(d.geom_xpos[ids]-p[t],axis=-1))))
        r_error=max(r_error,float(np.max(abs(d.geom_xmat[ids].reshape(k,3,3)-rot[t]))))
        actual=d.xpos[wrists]+np.einsum('nij,nj->ni',d.xmat[wrists].reshape(2,3,3),locals_[t])
        task_error=max(task_error,float(np.max(np.linalg.norm(actual-task[t],axis=-1))))
    report=dict(frames=n,body_visual_geoms=k,scalar_joints=checked_joints,
                visual_position_max_m=p_error,visual_rotation_matrix_max_abs=r_error,
                native_joint_max_abs=joint_error,tracking_task_position_max_m=task_error,
                coordinate_policy='Fixed declared display-to-native rotation only; no fitting, translation alignment or frame offset',
                kinematic_only=True)
    if p_error>2e-6 or r_error>2e-6 or joint_error>1e-8 or task_error>2e-6:
        raise ValueError('Native qpos / Grid playback parity failed: '+json.dumps(report))
    return report
