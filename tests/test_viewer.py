import io,json,zipfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from greenwich_pipelines.viewer.uploads import unpack_robot
from greenwich_pipelines.viewer.server import create_app
from greenwich_pipelines.viewer.export import C,motion_payload

URDF=b'''<robot name="Test arm"><link name="base"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link><link name="arm"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.1 0.1 0.5"/></geometry><origin xyz="0 0 .25"/></visual></link><joint name="lift" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 .2"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>'''

@pytest.mark.parametrize('path',['../escape.urdf','/escape.urdf','C:/escape.urdf','a/../../escape.urdf'])
def test_upload_rejects_archive_escape(tmp_path,path):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr(path,URDF)
    with pytest.raises(ValueError):unpack_robot(stream.getvalue(),'robot.zip',tmp_path/'uploaded')
    assert not (tmp_path/'escape.urdf').exists()

def test_uploaded_robot_loadable_is_not_conversion_certificate(tmp_path):
    config=tmp_path/'configs/viewer.local.json';config.parent.mkdir()
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[],clips=[])))
    with TestClient(create_app(config)) as client:
        result=client.post('/api/robots/locomotion',files={'file':('arm.urdf',URDF,'application/xml')})
        assert result.status_code==200
        robot=result.json();assert robot['conversion_ready'] is False
        assert robot['report']['links']==2 and robot['model_url'].endswith('.glb')
        assert client.get(robot['model_url']).status_code==200
        assert len(client.get('/api/catalog/locomotion').json()['robots'])==1
        assert client.get('/api/catalog/manipulation').json()['robots']==[]
        assert client.post('/api/convert',json={'clip_id':'missing','robot_id':robot['id']}).status_code==400
    with TestClient(create_app(config)) as client:assert len(client.get('/api/catalog/locomotion').json()['robots'])==1

def test_mesh_reference_cannot_read_local_absolute_path(tmp_path):
    data=b'<robot name="x"><link name="root"><visual><geometry><mesh filename="C:/private/file.stl"/></geometry></visual></link></robot>'
    with pytest.raises(ValueError):unpack_robot(data,'robot.urdf',tmp_path/'robot')

def test_descriptor_root_basis_is_conjugated_in_native_playback(tmp_path):
    import mujoco as mj
    from alphamotion.engine.descriptor import build_from_mjcf
    from scipy.spatial.transform import Rotation
    xml=tmp_path/'model.xml';xml.write_text('''<mujoco><worldbody><body name="base"><freejoint/><geom type="box" size=".1 .1 .1"/><body name="arm" pos="0 0 .4"><joint name="hinge" axis="0 1 0" range="-1 1"/><geom type="capsule" size=".04 .2"/></body></body></worldbody></mujoco>''')
    spec,_,_,_,_=build_from_mjcf(xml,'basis_test');J=spec.J
    R=Rotation.from_euler('y',30,degrees=True).as_matrix();rot=np.tile(np.r_[R[:,0],R[:,1]],(3,J,1));root=np.tile([12.,83.,-23.],(3,1))
    path=tmp_path/'motion.npz';np.savez(path,q=np.zeros((3,J,3)),rot6d=rot,root_t=root,joint_names=spec.joint_names,world_position_cm=np.tile(root[:,None,:],(1,J,1)))
    payload=motion_payload(dict(xml=str(xml),body='basis_test'),path,'fullbody')
    model=mj.MjModel.from_xml_path(str(xml));data=mj.MjData(model);data.qpos[:3]=C.T@root[0]/100;data.qpos[3:7]=Rotation.from_matrix(C.T@R@C).as_quat(scalar_first=True);mj.mj_forward(model,data)
    assert np.allclose(payload['positions'][0],data.geom_xpos[payload['geom_ids']]@C.T,atol=1e-7)
    actual=Rotation.from_quat(payload['quaternions'][0]).as_matrix()
    assert np.allclose(actual,C@data.geom_xmat[payload['geom_ids']].reshape(-1,3,3),atol=1e-7)
