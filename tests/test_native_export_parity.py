"""Native tool offsets must not inherit a merged descriptor's distal frame."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

mj=pytest.importorskip('mujoco')
from arenalab_workbench.native_export_parity import audit_native_export,native_task_points

BASIS=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
XML='''<mujoco><worldbody><body name="base"><freejoint/>
<geom type="box" size=".1 .1 .1" group="2"/>
<body name="left_wrist" pos="0 .2 .1"><joint name="left_pitch" axis="0 1 0"/>
<geom type="sphere" size=".02" group="2"/>
<body name="left_distal" euler="0 35 0"><geom type="sphere" size=".01" group="2"/></body></body>
<body name="right_wrist" pos="0 -.2 .1"><joint name="right_pitch" axis="0 1 0"/>
<geom type="sphere" size=".02" group="2"/>
<body name="right_distal" euler="0 -35 0"><geom type="sphere" size=".01" group="2"/></body></body>
</body></worldbody></mujoco>'''


def test_distal_descriptor_frames_do_not_rotate_native_tcp_offsets(tmp_path):
    xml=tmp_path/'model.xml';xml.write_text(XML)
    model=mj.MjModel.from_xml_path(str(xml));state=mj.MjData(model)
    q=np.tile(model.qpos0,(4,1));q[:,:3]=[[0,0,1],[.2,.1,1.1],[.1,-.2,1.2],[.4,.3,.9]]
    q[:,3:7]=Rotation.from_euler('xyz',[[0,0,0],[.1,.2,.3],[-.2,.1,.4],[.3,-.2,-.1]]).as_quat(scalar_first=True)
    q[:,7:]=[[0,0],[.2,-.1],[-.3,.4],[.5,-.2]]
    wrists=[model.body(s+'_wrist').id for s in ('left','right')]
    distal=[model.body(s+'_distal').id for s in ('left','right')]
    positions=[];rotations=[];wrist_positions=[];wrist_rotations=[];descriptor_rotations=[]
    for pose in q:
        state.qpos[:]=pose;mj.mj_forward(model,state)
        positions.append(state.geom_xpos.copy());rotations.append(state.geom_xmat.reshape(-1,3,3).copy())
        wrist_positions.append(state.xpos[wrists].copy());wrist_rotations.append(state.xmat[wrists].reshape(2,3,3).copy())
        descriptor_rotations.append(state.xmat[distal].reshape(2,3,3).copy())
    motion=dict(wrist_body_position_native_m=np.asarray(wrist_positions),wrist_body_rotation_native=np.asarray(wrist_rotations))
    local=np.broadcast_to([0.,0.,-.09],(4,2,7,3)).copy()
    local[:,:,1:,0]=np.arange(1,7)*.01
    actual=native_task_points(motion,local)
    # This is the old failure: equal wrist origins do not imply equal frames.
    legacy=motion['wrist_body_position_native_m']+np.einsum('tsij,tsj->tsi',descriptor_rotations,local[:,:,0])
    assert np.linalg.norm(legacy-actual[:,:,0],axis=-1).min()>.05
    payload=dict(frames=4,fps=30,positions=(np.asarray(positions)@BASIS.T).tolist(),
        quaternions=Rotation.from_matrix((BASIS@np.asarray(rotations)).reshape(-1,3,3)).as_quat().reshape(4,-1,4).tolist(),
        tracking={'actual':(actual[:,:,0]@BASIS.T).tolist()},
        joints={'names':['left_pitch','right_pitch'],'values':q[:,7:].tolist()})
    qpath=tmp_path/'qpos.npz';np.savez(qpath,qpos=q)
    render=tmp_path/'render.npz';np.savez(render,native_body_geom_ids=np.arange(model.ngeom))
    playback=tmp_path/'playback.json';playback.write_text(json.dumps(payload))
    row=dict(xml=str(xml),endpoints=dict(left_wrist='left_wrist',right_wrist='right_wrist'))
    report=audit_native_export(row,qpath,playback,render,local)
    assert report['tracking_task_position_max_m']<1e-12
    assert actual.shape==(4,2,7,3)  # palm and all fingertip/interaction points
    payload['tracking']['actual']=(legacy@BASIS.T).tolist();playback.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='Native qpos / Grid playback parity failed'):
        audit_native_export(row,qpath,playback,render,local)


@pytest.mark.parametrize('failure',['missing','bad_shape','nan','reflection'])
def test_native_frame_metadata_is_required_and_validated(failure):
    motion=dict(wrist_body_position_native_m=np.zeros((2,2,3)),wrist_body_rotation_native=np.tile(np.eye(3),(2,2,1,1)))
    local=np.zeros((2,2,1,3))
    if failure=='missing':motion.pop('wrist_body_rotation_native')
    elif failure=='bad_shape':local=local[:1]
    elif failure=='nan':motion['wrist_body_position_native_m'][0,0,0]=np.nan
    elif failure=='reflection':motion['wrist_body_rotation_native'][0,0,0,0]=-1
    with pytest.raises((ValueError,KeyError)):
        native_task_points(motion,local)
