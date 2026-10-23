"""Native FK preview integrity, revision isolation and workspace boundaries."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import mujoco as mj
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from arenalab_workbench import native_source_preview as preview


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    from arenalab_workbench import settings,source_identity,robot_motion_source,gpu_scheduler
    from arenalab_workbench.dataset_io import storage
    from arenalab_motion import prior
    store=tmp_path/'store';store.mkdir();work=tmp_path/'work';work.mkdir()
    monkeypatch.setattr(settings.paths,'store',store);monkeypatch.setattr(settings.paths,'workbench',work)
    monkeypatch.setattr(storage,'STORE',store);monkeypatch.setattr(storage,'ROOT',work)
    monkeypatch.setattr(gpu_scheduler,'ROOT',tmp_path/'scheduler')
    xml=tmp_path/'source.xml'
    xml.write_text('''<mujoco><worldbody><body name="base"><freejoint/><geom type="box" size=".1 .2 .3"/>
      <body name="arm" pos=".1 .2 .3"><joint name="shoulder" axis="0 1 0"/>
        <geom type="capsule" size=".03 .2"/><body name="elbow" pos="0 0 .4">
          <joint name="elbow" axis="1 0 0"/><geom type="box" size=".04 .05 .08"/></body>
      </body></body></worldbody></mujoco>''')
    robot=dict(xml=str(xml),body='native-test')
    monkeypatch.setattr(robot_motion_source,'source_robot',lambda spec:robot.copy())
    monkeypatch.setattr(prior,'load_target',lambda row:dict(model=mj.MjModel.from_xml_path(row['xml'])))
    monkeypatch.setattr(robot_motion_source,'prepare_robot_source',lambda *a,**kw:pytest.fail('Preview ran full source adaptation'))
    times=7+np.arange(5)/30
    raw=dict(timestamps_s=times,joint_names=np.array(['elbow','shoulder']),
        joint_positions=np.array([[i*.08,-i*.1] for i in range(5)]),root_position=np.array([[1+i*.1,2.,.8] for i in range(5)]),
        root_quaternion_wxyz=Rotation.from_euler('xyz',[[.2,.1,i*.2] for i in range(5)]).as_quat(scalar_first=True))
    state=tmp_path/'motion.npz';np.savez(state,**raw)
    objects=tmp_path/'objects.npz'
    object_data=dict(object_ids=np.array(['cube']),timestamps_s=times,
        positions_world_m=np.array([[[2.,1.,.4+i*.02]] for i in range(5)]),
        rotations_world=Rotation.from_euler('z',np.arange(5)*.15).as_matrix()[:,None],
        object_0_vertices_local_m=np.array([[0.,0,0],[.2,0,0],[0,.2,0]]),object_0_faces=np.array([[0,1,2]],np.int32))
    np.savez(objects,**object_data)
    spec=dict(format='robot_joint_state',family='robot',robot_id='local-source-only',path=str(state),objects_path=str(objects),
        sha256=preview._sha(state),floor_z_m=0.,units='m',world_up='z',evidence='retargeted',source_stage='retargeted',video_role='preview-only')
    row=dict(id='clip--g1',source=str(state),frames=5,fps=30.,motion_source=spec)
    monkeypatch.setattr(source_identity,'registered_source',lambda sid:(row.copy(),spec.copy()))
    return SimpleNamespace(root=tmp_path,store=store,work=work,xml=xml,robot=robot,raw=raw,spec=spec,row=row,
        state=state,objects=objects,object_data=object_data,times=times)


def test_native_mesh_fk_matches_fresh_mujoco_all_frames_without_human_or_target(fixture):
    f=fixture;result=preview.source_preview('clip--g1');p=result['source_motion']
    assert result['cache_hit'] is False and result['recording_id']=='clip--g1'
    assert p['source_family']=='robot' and p['source_robot_id']=='local-source-only'
    assert p['retargeted'] is False and p['source_trajectory_modified'] is False
    assert p['source_evidence']==p['source_stage']=='retargeted'
    assert p['preview_retargeting_performed'] is False
    assert p['frames']==5 and p['fps']==30 and p['world_up']=='z'
    payload=json.loads((f.store/p['playback_url'].removeprefix('/assets/')).read_text())
    from greenwich_pipelines.viewer.export import visible_geoms
    model=mj.MjModel.from_xml_path(str(f.xml));data=mj.MjData(model);ids=visible_geoms(model)
    for t in range(5):
        data.qpos[:3]=f.raw['root_position'][t];data.qpos[3:7]=f.raw['root_quaternion_wxyz'][t]
        for i,name in enumerate(f.raw['joint_names']):data.qpos[model.jnt_qposadr[model.joint(name).id]]=f.raw['joint_positions'][t,i]
        mj.mj_forward(model,data)
        np.testing.assert_allclose(payload['positions'][t],data.geom_xpos[ids],atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(payload['quaternions'][t]).as_matrix(),data.geom_xmat[ids].reshape(-1,3,3),atol=1e-14)
    obj=json.loads((f.store/p['objects'][0]['poses_url'].removeprefix('/assets/')).read_text())
    np.testing.assert_array_equal(obj['timestamps_s'],f.times)
    np.testing.assert_array_equal(obj['position'],f.object_data['positions_world_m'][:,0])
    assert not list(f.work.rglob('smpl*')) and not list(f.store.rglob('*.xml'))
    assert str(f.root) not in json.dumps(result)


def test_second_request_verifies_and_reuses_cached_preview(fixture,monkeypatch):
    first=preview.source_preview('clip--g1')
    monkeypatch.setattr(preview,'_render',lambda *a:pytest.fail('Cache hit ran FK'))
    second=preview.source_preview('clip--g1')
    assert second['cache_hit'] is True and first['source_motion']==second['source_motion']


@pytest.mark.parametrize('mutation',['source','objects','model','code'])
def test_change_source_object_model_or_decoder_never_hits_old_preview(fixture,monkeypatch,mutation):
    f=fixture;first=preview.source_preview('clip--g1')
    if mutation=='source':
        f.raw['root_position'][:,0]+=.3;np.savez(f.state,**f.raw);f.spec['sha256']=preview._sha(f.state)
    elif mutation=='objects':
        f.object_data['positions_world_m'][:,0,0]+=.2;np.savez(f.objects,**f.object_data)
    elif mutation=='model':f.xml.write_text(f.xml.read_text().replace('size=".1 .2 .3"','size=".2 .2 .3"'))
    else:
        original=preview._identity
        monkeypatch.setattr(preview,'_identity',lambda spec,row:{**original(spec,row),'decoder_revision':'changed'})
    second=preview.source_preview('clip--g1')
    assert second['cache_hit'] is False
    assert second['source_motion']['model_url']!=first['source_motion']['model_url']
    assert len(list(f.work.glob('source-preview-cache/public/*/receipt.json')))==2


@pytest.mark.parametrize('target',['cache','model','objects'])
def test_corrupt_cache_or_asset_fails_instead_of_false_hit(fixture,target):
    f=fixture;r=preview.source_preview('clip--g1')
    if target=='cache':p=next(f.work.glob('source-preview-cache/public/*/source/source_robot_playback.npz'))
    elif target=='model':p=f.store/r['source_motion']['model_url'].removeprefix('/assets/')
    else:p=f.store/r['source_motion']['objects'][0]['poses_url'].removeprefix('/assets/')
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed|integrity'):preview.source_preview('clip--g1')


def test_private_assets_are_marked_and_never_share_public_namespace(fixture):
    from arenalab_workbench.dataset_io import storage
    f=fixture;token='owner-token';owner=storage.owner(token)
    private=preview.source_preview('clip--g1',workspace=owner)
    p=f.store/private['source_motion']['model_url'].removeprefix('/assets/')
    assert storage.permitted(p,token) and not storage.permitted(p,'another') and not storage.permitted(p,None)
    public=preview.source_preview('clip--g1')
    q=f.store/public['source_motion']['model_url'].removeprefix('/assets/')
    assert q!=p and storage.permitted(q,None)
    assert preview.source_preview('clip--g1',workspace=owner)['cache_hit']


def test_concurrent_same_source_builds_exactly_once(fixture,monkeypatch):
    original=preview._render;calls=[];guard=threading.Lock()
    def render(*args):
        with guard:calls.append(1)
        return original(*args)
    monkeypatch.setattr(preview,'_render',render)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(preview.source_preview,['clip--g1','clip--g1']))
    assert len(calls)==1 and sorted(r['cache_hit'] for r in results)==[False,True]


def test_changed_source_during_build_does_not_publish_cache(fixture,monkeypatch):
    f=fixture;original=preview._render
    def render(*args):
        original(*args);f.object_data['positions_world_m'][:,0,0]+=.5;np.savez(f.objects,**f.object_data)
    monkeypatch.setattr(preview,'_render',render)
    with pytest.raises(ValueError,match='changed during preview'):preview.source_preview('clip--g1')
    assert not list(f.work.glob('source-preview-cache/*/*/receipt.json'))


def test_mismatched_object_clock_rejected_without_slicing(fixture):
    f=fixture;f.object_data['timestamps_s']=f.times+1/30;np.savez(f.objects,**f.object_data)
    with pytest.raises(ValueError,match='exact source motion clock'):preview.source_preview('clip--g1')
    assert not list(f.work.glob('source-preview-cache/*/*/receipt.json'))


def test_route_session_private_owner_and_no_path_parameters(fixture,monkeypatch):
    from arenalab_workbench import height_registry
    from arenalab_workbench.dataset_io import routes,storage
    f=fixture;owner=storage.owner('mine');public=dict(id='clip--g1',owner=owner,source_preview_available=True)
    storage.write(storage.ROOT/'user-data/registry.json',{'clip--g1':dict(public=public)})
    monkeypatch.setattr(height_registry,'public_entry',lambda sid:public if sid=='clip--g1' else None)
    app=FastAPI();routes.attach_dataset_routes(app,threading.Lock());client=TestClient(app)
    assert client.post('/source-preview/clip--g1').status_code==401
    client.cookies.set(storage.COOKIE,'other')
    assert client.post('/source-preview/clip--g1').status_code==404
    client.cookies.set(storage.COOKIE,'mine')
    assert client.post('/source-preview/missing').status_code==404
    result=client.post('/source-preview/clip--g1',json={'path':'/other/source.npz','xml':'/other/model.xml'})
    assert result.status_code==200,result.text
    assert result.json()['source_motion']['source_sha256']==f.spec['sha256']
    assert str(f.root) not in result.text
    p=f.store/result.json()['source_motion']['model_url'].removeprefix('/assets/')
    assert storage.permitted(p,'mine') and not storage.permitted(p,'other')


def test_route_does_not_leak_model_paths_on_failure(fixture,monkeypatch):
    from arenalab_workbench import height_registry
    from arenalab_workbench.dataset_io import routes,storage
    monkeypatch.setattr(height_registry,'public_entry',lambda sid:dict(id=sid,owner=None))
    monkeypatch.setattr(preview,'source_preview',lambda *a,**k:(_ for _ in ()).throw(ValueError('/private/customer/model.xml')))
    app=FastAPI();routes.attach_dataset_routes(app,threading.Lock());client=TestClient(app);client.cookies.set(storage.COOKIE,'mine')
    response=client.post('/source-preview/clip--g1')
    assert response.status_code==409 and '/private/' not in response.text


def test_legacy_robot_preview_paths_unchanged_until_explicit_source_annotations(fixture):
    import hashlib
    from arenalab_workbench.source_preview import export_source_preview
    f=fixture;source=f.root/'legacy-source';spec=dict(f.spec);spec.pop('evidence');spec.pop('source_stage')
    preview._render(spec,source,f.robot)
    provenance_path=source/'source_provenance.json';provenance=json.loads(provenance_path.read_text())
    identity=dict(motion=preview._sha(source/'source_robot_playback.npz'),model=preview._sha(source/'source_robot.glb'),
        objects=preview._sha(source/'objects.npz'),video_role='preview-only')
    old_key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    before=export_source_preview(source,f.store/'legacy')
    assert '/source-robot/'+old_key+'/' in before['model_url']
    assert 'source_evidence' not in before and 'preview_retargeting_performed' not in before
    old_manifest=f.store/before['model_url'].removeprefix('/assets/');old_manifest=old_manifest.with_name('manifest.json')
    old_bytes=old_manifest.read_bytes()
    provenance.update(source_evidence='retargeted',source_stage='retargeted');provenance_path.write_text(json.dumps(provenance))
    after=export_source_preview(source,f.store/'legacy')
    assert after['model_url']!=before['model_url']
    assert after['source_evidence']==after['source_stage']=='retargeted' and after['preview_retargeting_performed'] is False
    assert old_manifest.read_bytes()==old_bytes
    assert export_source_preview(source,f.store/'legacy')==after


def test_private_preview_asset_http_access_is_owner_scoped(fixture):
    from arena_api.media import attach_media
    from arenalab_workbench.dataset_io import storage
    f=fixture;result=preview.source_preview('clip--g1',workspace=storage.owner('mine'))
    app=FastAPI();attach_media(app,f.store);client=TestClient(app)
    relative=result['source_motion']['model_url'].removeprefix('/assets/')
    assert client.get('/media/'+relative).status_code==404
    client.cookies.set(storage.COOKIE,'other')
    assert client.get('/media/'+relative).status_code==404
    client.cookies.set(storage.COOKIE,'mine')
    response=client.get('/media/'+relative)
    assert response.status_code==200 and response.content[:4]==b'glTF'
