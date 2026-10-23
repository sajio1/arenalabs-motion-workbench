import io,json,zipfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from greenwich_pipelines.viewer.uploads import unpack_robot
from greenwich_pipelines.viewer.server import create_app
from greenwich_pipelines.viewer.export import C,motion_payload

URDF=b'''<robot name="Test arm"><link name="base"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link><link name="arm"><inertial><mass value="1"/><inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.1 0.1 0.5"/></geometry><origin xyz="0 0 .25"/></visual></link><joint name="lift" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 .2"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>'''

def test_two_clients_share_worker_but_second_reports_queue_position(tmp_path,monkeypatch):
    import threading,time
    from types import SimpleNamespace
    import greenwich_pipelines.viewer.server as server
    entered=[threading.Event(),threading.Event()];release=[threading.Event(),threading.Event()];calls=[]
    def blocked(*args,**kwargs):
        i=len(calls);calls.append(i);entered[i].set()
        assert release[i].wait(10)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(server,'run_conversion',blocked)
    monkeypatch.setattr(server,'motion_preflight',lambda clip,robot:{'morphology':{'available':False}})
    monkeypatch.setattr(server,'conversion_signature',lambda *args:'fixed-test-signature')
    config=tmp_path/'configs/viewer.json';config.parent.mkdir()
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[dict(id='r',pages=['locomotion'],conversion_ready=True,pipeline_row={})],clips=[dict(id=c,page='locomotion') for c in ['a','b']])))
    app=create_app(config)
    try:
        with TestClient(app) as first,TestClient(app) as second:
            one=first.post('/api/convert',json=dict(clip_id='a',robot_id='r')).json()
            assert entered[0].wait(3)
            two=second.post('/api/convert',json=dict(clip_id='b',robot_id='r')).json()
            assert two['status']=='queued' and two['queue_position']==1
            assert first.get('/api/jobs/'+one['id']).json()['status']=='running'
            assert second.post('/api/convert',json=dict(clip_id='b',robot_id='r')).json()['id']==two['id']
            assert not entered[1].is_set()
            release[0].set();assert entered[1].wait(3)
            current=second.get('/api/jobs/'+two['id']).json()
            assert current['status']=='running' and 'queue_position' not in current
    finally:
        for event in release:event.set()
        app.state.executor.shutdown(wait=True)

def test_catalog_hides_retired_choice_and_exposes_explicit_presets(tmp_path):
    config=tmp_path/'configs/viewer.json';config.parent.mkdir()
    presets=[dict(clip_id='dance',robot_id='h2')]
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[dict(id=r,pages=['locomotion']) for r in ['h2','retired']],clips=[],hidden_robot_ids=['retired'],presets=dict(locomotion=presets))))
    with TestClient(create_app(config)) as client:
        result=client.get('/api/catalog/locomotion').json()
        assert [r['id'] for r in result['robots']]==['h2']
        assert result['presets']==presets

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

def test_catalog_preserves_source_specific_nested_metadata(tmp_path):
    config=tmp_path/'configs/viewer.local.json';config.parent.mkdir()
    metadata={'Capture device':'Xsens','Contacts':{'heel':12},'Channels':['wrist','ankle']}
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[],clips=[dict(id='a',page='localmanipulation',metadata=metadata,source='private/path')])) )
    with TestClient(create_app(config)) as client:
        clip=client.get('/api/catalog/localmanipulation').json()['clips'][0]
        assert clip['metadata']==metadata
        assert 'source' not in clip

def test_long_recording_playback_keeps_all_windows_and_boundaries():
    from greenwich_pipelines.viewer.export import join_playback_segments
    def segment(values):
        return dict(fps=30,frames=len(values),geom_ids=[0],parents=[-1],unit='m',axis='y-up',positions=list(values),quaternions=list(values),skeleton=list(values),targets=list(values))
    result=join_playback_segments([segment([0,1,2]),segment([3,4])])
    assert result['frames']==5 and result['duration']==4/30
    assert result['positions']==[0,1,2,3,4]
    assert result['window_boundaries']==[0,3]
    other=segment([3]);other['geom_ids']=[2]
    with pytest.raises(ValueError):join_playback_segments([segment([0]),other])

def test_source_window_preserves_endpoint_positions_rotations_and_contact(tmp_path):
    from greenwich_pipelines.viewer.sources import prepare_source
    frames=40;positions=np.arange(frames*4*3,dtype=np.float32).reshape(frames,4,3)
    rotations=np.tile(np.eye(3,dtype=np.float32),(frames,4,1,1));contact=np.arange(frames*4).reshape(frames,4)%2==0
    source=tmp_path/'sparse.npz';np.savez(source,position_yup_cm=positions,semantic_rotation_yup=rotations,foot_contacts=contact,fps=np.array(30.))
    clip=dict(page='localmanipulation',source=str(source),source_window=dict(start_frame=5,end_frame=35))
    output=prepare_source(clip,tmp_path)
    with np.load(output) as d:
        assert np.array_equal(d['position_yup_cm'],positions[5:35])
        assert np.array_equal(d['semantic_rotation_yup'],rotations[5:35])
        assert np.array_equal(d['foot_contacts'],contact[5:35])
        assert np.allclose(np.diff(d['sample_timestamp_us']),1e6/30)
        assert float(d['fps'])==30
    clip['source_window']['end_frame']=45
    with pytest.raises(ValueError):prepare_source(clip,tmp_path)


def test_source_video_registry_and_range_seek(tmp_path):
    config=tmp_path/'configs/viewer.local.json';config.parent.mkdir()
    video=tmp_path/'source.mp4';video.write_bytes(bytes(range(100)))
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[],clips=[dict(id='known',page='manipulation',video=dict(path=str(video),start_seconds=2,sync_method='paired'))])))
    with TestClient(create_app(config)) as client:
        row=client.get('/api/catalog/manipulation').json()['clips'][0]
        assert row['video']['url']=='/api/source-video/known'
        assert 'path' not in row['video']
        response=client.get(row['video']['url'],headers={'Range':'bytes=10-19'})
        assert response.status_code==206 and response.content==bytes(range(10,20))
        assert client.get('/api/source-video/missing').status_code==404


def test_nested_parent_mesh_reference_and_compiler_paths(tmp_path):
    import trimesh
    stream=io.BytesIO()
    xml=URDF.replace(b'<box size="0.2 0.2 0.2"/>',b'<mesh filename="../meshes/base.stl"/>')
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('package/urdf/model.urdf',xml)
        z.writestr('package/meshes/base.stl',trimesh.creation.box().export(file_type='stl'))
    path,_,report=unpack_robot(stream.getvalue(),'robot.zip',tmp_path/'uploaded')
    assert not report['missing_meshes']
    import mujoco as mj
    assert mj.MjModel.from_xml_path(str(path)).ngeom==2


def test_mesh_reference_cannot_escape_extracted_package(tmp_path):
    bad=URDF.replace(b'<box size="0.2 0.2 0.2"/>',b'<mesh filename="../../secret.stl"/>')
    with pytest.raises(ValueError,match='inside the robot package'):
        unpack_robot(bad,'robot.urdf',tmp_path/'uploaded')


def test_humanoid_upload_creates_shared_persisted_descriptor(tmp_path):
    import xml.etree.ElementTree as ET
    root=ET.Element('robot',name='generic')
    for name in ['pelvis','wrist_l_link','wrist_r_link','ankle_l_link','ankle_r_link']:
        link=ET.SubElement(root,'link',name=name)
        inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='1');ET.SubElement(inertial,'inertia',ixx='.01',iyy='.01',izz='.01',ixy='0',ixz='0',iyz='0')
        visual=ET.SubElement(link,'visual');ET.SubElement(ET.SubElement(visual,'geometry'),'box',size='.1 .05 .1')
    for name,pos in [('wrist_l_link','0 .3 .1'),('wrist_r_link','0 -.3 .1'),('ankle_l_link','0 .1 -.5'),('ankle_r_link','0 -.1 -.5')]:
        joint=ET.SubElement(root,'joint',name=name+'_joint',type='revolute');ET.SubElement(joint,'parent',link='pelvis');ET.SubElement(joint,'child',link=name);ET.SubElement(joint,'origin',xyz=pos);ET.SubElement(joint,'axis',xyz='0 1 0');ET.SubElement(joint,'limit',lower='-1',upper='1',effort='10',velocity='2')
    config=tmp_path/'configs/viewer.local.json';config.parent.mkdir()
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[],clips=[])))
    with TestClient(create_app(config)) as client:
        row=client.post('/api/robots/localmanipulation',files={'file':('generic.urdf',ET.tostring(root),'application/xml')}).json()
        assert row['conversion_ready'] and row['status']=='ready'
        assert set(row['pages'])=={'locomotion','localmanipulation'}
        assert len(row['report']['endpoints'])==4
        assert row['report']['floating_root_added']
        bad=dict(endpoints=row['report']['endpoints'],palm_frames={'left_wrist':{'distal':[1,0,0],'normal':[1,0,0]},'right_wrist':{'distal':[1,0,0],'normal':[0,1,0]}})
        assert client.patch('/api/robots/'+row['id']+'/metadata',json=bad).status_code==400
        valid=dict(endpoints=row['report']['endpoints'],palm_frames=row['report']['palm_frames'])
        assert client.patch('/api/robots/'+row['id']+'/metadata',json=valid).status_code==200
        folder=tmp_path/'store/uploaded'/row['id']
        descriptor=json.loads((folder/'descriptor.json').read_text())
        import mujoco as mj
        model=mj.MjModel.from_xml_path(descriptor['xml'])
        assert sum(model.jnt_type==mj.mjtJoint.mjJNT_FREE)==1
        from alphamotion.engine.descriptor import build_from_mjcf
        from greenwich_pipelines.pipelines.nymeria.input import endpoint_indices
        spec,*_=build_from_mjcf(descriptor['xml'],descriptor['body'])
        assert [spec.joint_names[i] for i in endpoint_indices(spec,descriptor)]==list(descriptor['endpoints'].values())
    with TestClient(create_app(config)) as client:
        assert client.get('/api/catalog/locomotion').json()['robots'][0]['conversion_ready']

    from greenwich_pipelines.viewer.server import conversion_signature
    saved=json.loads(config.read_text());source=tmp_path/'source.npz';source.write_bytes(b'unchanged source')
    clip=dict(id='source',page='localmanipulation',source=str(source))
    saved['clips']=[clip];config.write_text(json.dumps(saved))
    registry=json.loads((tmp_path/'store/uploaded/registry.json').read_text());robot=registry[0]
    output=tmp_path/'store/jobs/cached/playback.json';output.parent.mkdir(parents=True);output.write_text('{}')
    cache=dict(clip_id='source',robot_id=robot['id'],signature=conversion_signature(clip,robot,runtime='unused'),url='/assets/jobs/cached/playback.json')
    (tmp_path/'store/viewer-conversions.json').write_text(json.dumps({'entry':cache}))
    with TestClient(create_app(config)) as client:
        response=client.post('/api/convert',json=dict(clip_id='source',robot_id=robot['id'])).json()
        assert response['cached'] and response['status']=='completed'
    source.write_bytes(b'changed source')
    with TestClient(create_app(config)) as client:
        assert robot['id'] not in client.get('/api/catalog/localmanipulation').json()['clips'][0].get('cache',{})


def test_shoulders_resolve_from_tree_without_shoulder_names():
    from types import SimpleNamespace
    from greenwich_pipelines.robot_semantics import shoulder_landmarks
    spec=SimpleNamespace(parents=np.array([-1,0,1,2,3,1,5,6]),joint_names=['base','torso','left_upper_arm','left_forearm','left_end_effector','right_upper_arm','right_forearm','right_end_effector'])
    assert shoulder_landmarks(spec,[4,7])==[2,5]


def test_failed_conversion_reports_reason_and_task_id(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    import greenwich_pipelines.viewer.server as server
    def failed(*args,**kwargs):
        kwargs['stdout'].write('ValueError: Animated non-root translation at Spine2\n')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(server,'run_conversion',failed)
    monkeypatch.setattr(server,'motion_preflight',lambda clip,robot:{'morphology':{'available':False}})
    monkeypatch.setattr(server,'conversion_signature',lambda *args:'fixed-test-signature')
    config=tmp_path/'configs/viewer.json';config.parent.mkdir()
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',
        robots=[dict(id='r',pages=['locomotion'],conversion_ready=True,pipeline_row={})],
        clips=[dict(id='a',page='locomotion')])))
    app=create_app(config)
    try:
        with TestClient(app) as client:
            job=client.post('/api/convert',json=dict(clip_id='a',robot_id='r')).json()
            for _ in range(100):
                result=client.get('/api/jobs/'+job['id']).json()
                if result['status']=='failed':break
                time.sleep(.01)
            assert result['status']=='failed'
            assert 'Animated non-root translation at Spine2' in result['error']
            assert job['id'] in result['error']
            assert 'local job directory' not in result['error']
    finally:app.state.executor.shutdown(wait=True)


def test_four_gpu_workers_are_isolated_and_overflow_stays_queued(tmp_path,monkeypatch):
    import threading,time
    from types import SimpleNamespace
    import greenwich_pipelines.viewer.server as server
    release=threading.Event();started=[];guard=threading.Lock()
    def run(*args,**kwargs):
        with guard:started.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
        assert release.wait(10)
        kwargs['stdout'].write('RuntimeError: test worker finished\n')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(server,'run_conversion',run)
    monkeypatch.setattr(server,'motion_preflight',lambda clip,robot:{'morphology':{'available':False}})
    monkeypatch.setattr(server,'conversion_signature',lambda *args:'fixed-test-signature')
    config=tmp_path/'configs/viewer.json';config.parent.mkdir()
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',conversion_devices=['0','1','2','3'],
        robots=[dict(id='r',pages=['locomotion'],conversion_ready=True,pipeline_row={})],
        clips=[dict(id=str(i),page='locomotion') for i in range(5)])))
    app=create_app(config)
    try:
        with TestClient(app) as client:
            jobs=[client.post('/api/convert',json=dict(clip_id=str(i),robot_id='r')).json() for i in range(5)]
            deadline=time.monotonic()+5
            while len(started)<4 and time.monotonic()<deadline:time.sleep(.01)
            assert sorted(started)==['0','1','2','3']
            states=[client.get('/api/jobs/'+j['id']).json() for j in jobs]
            assert [s['status'] for s in states]==['running']*4+['queued']
            assert states[-1]['queue_position']==1
            duplicate=client.post('/api/convert',json=dict(clip_id='4',robot_id='r')).json()
            assert duplicate['id']==jobs[-1]['id']
            release.set()
            app.state.executor.shutdown(wait=True)
            assert len(started)==5 and started[-1] in {'0','1','2','3'}
            assert all(client.get('/api/jobs/'+j['id']).json()['status']=='failed' for j in jobs)
    finally:
        release.set();app.state.executor.shutdown(wait=True)


def test_unified_worker_scale_modes_are_isolated_and_reused(tmp_path,monkeypatch):
    import time
    from types import SimpleNamespace
    import greenwich_pipelines.viewer.server as server
    calls=[]
    monkeypatch.setattr(server,'motion_preflight',lambda clip,robot:{'scale_choice_required':True,'morphology':{'available':True,'large_mismatch':True},'source_policy':'provided-fullbody','source_policy_version':'test-source'})
    monkeypatch.setattr(server,'conversion_signature',lambda clip,robot,mode='preserve',runtime=None:'signature:'+mode)
    def run(command,**kwargs):
        assert command[1:4]==['-m','arenalab_workbench.unified_conversion','--request']
        request=json.loads(Path(command[4]).read_text());calls.append(request)
        output=Path(request['output']);output.mkdir()
        (output/'playback.json').write_text(json.dumps(dict(fps=30,frames=3,positions=[],quaternions=[])))
        (output/'report.json').write_text(json.dumps(dict(solver_status='VALID_STOPPED',solver_converged=False)))
        assert kwargs['env']['ARENA_TRAJECTORY_SCALE']==request['trajectory_scale']
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(server,'run_conversion',run)
    config=tmp_path/'configs/viewer.json';config.parent.mkdir()
    config.write_text(json.dumps(dict(storage=str(tmp_path/'store'),runtime='unused',robots=[dict(id='r',pages=['locomotion'],conversion_ready=True,pipeline_row={})],clips=[dict(id='a',page='locomotion')])))
    app=create_app(config)
    try:
        with TestClient(app) as client:
            assert client.post('/api/convert',json=dict(clip_id='a',robot_id='r')).status_code==409
            assert not calls
            urls=[]
            for mode in ['preserve','match_robot']:
                body=dict(clip_id='a',robot_id='r',trajectory_scale=mode)
                response=client.post('/api/convert',json=body);assert response.status_code==200
                job=response.json()
                for _ in range(100):
                    value=client.get('/api/jobs/'+job['id']).json()
                    if value['status'] in {'completed','failed'}:break
                    time.sleep(.01)
                assert value['status']=='completed',value
                assert value['trajectory_scale']==mode and value['solver_converged'] is False
                urls.append(value['url'])
                again=client.post('/api/convert',json=body).json()
                assert again['cached'] and again['url']==urls[-1] and again['trajectory_scale']==mode
            assert urls[0]!=urls[1] and len(calls)==2
            catalog=client.get('/api/catalog/locomotion').json()
            assert catalog['clips'][0]['cache']['r']==urls[0]
    finally:app.state.executor.shutdown(wait=True)


def test_source_manifest_identity_follows_referenced_motion_and_bind_files(tmp_path):
    from greenwich_pipelines.viewer.server import source_content_identity
    manifest=tmp_path/'sources.json';motion=tmp_path/'motion.npz';options=tmp_path/'options.json';bind=tmp_path/'rest.json'
    motion.write_bytes(b'original motion');bind.write_text('{}');options.write_text(json.dumps({'bind_path':'rest.json'}))
    manifest.write_text(json.dumps({'sources':[{'path':'motion.npz','options_file':'options.json'}]}))
    clip={'source':str(manifest)}
    original=source_content_identity(clip)
    assert set(original)=={str(p) for p in (manifest,motion,options,bind)}
    motion.write_bytes(b'changed underlying motion')
    changed=source_content_identity(clip)
    assert original[str(manifest)]==changed[str(manifest)]
    assert original[str(motion)]!=changed[str(motion)]
    bind.write_text('{"neutral":"changed"}')
    assert changed[str(bind)]!=source_content_identity(clip)[str(bind)]
