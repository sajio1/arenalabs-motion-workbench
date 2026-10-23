"""Export selects immutable deliverables, including video-free robot datasets."""
import json
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from arenalab_workbench.dataset_io import routes, export as exporter
from arenalab_workbench.unified_scene import scene_key


def fixture_scene(root, outputs=None, mode='preserve', robot='g1', legacy=False, public=False):
    folder=root/('legacy' if public else 'manipulation4d/unified/'+scene_key('clip',mode,robot,None if legacy else outputs))
    folder.mkdir(parents=True,exist_ok=True)
    scene={'id':'clip','robots':[{'id':robot,'trajectory_scale':mode}],'generation_status':{'status':'completed','trajectory_scale':mode}}
    if not legacy:
        scene.update(output_modalities=outputs,completed_outputs=outputs,output_status={name:'completed' for name in outputs})
    path=folder/'scene.json';path.write_text(json.dumps(scene));return path,scene


@pytest.fixture
def isolated_store(tmp_path,monkeypatch):
    root=tmp_path/'store';root.mkdir()
    monkeypatch.setattr(routes.s,'STORE',root);monkeypatch.setattr(exporter,'STORE',root)
    caps={'outputs':{name:{'available':name in {'trajectory','point_cloud'}} for name in ('trajectory','point_cloud','contact','inpaint_fov')}}
    monkeypatch.setattr('arenalab_workbench.source_requirements.input_requirements',lambda *a,**k:{'output_capabilities':caps})
    return root


def test_omitted_outputs_explicit_scale_selects_new_default_variant_without_public_url(isolated_store):
    path,_=fixture_scene(isolated_store,['trajectory','point_cloud'],mode='match_robot')
    selected,_=routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',trajectory_scale='match_robot'),'clip',{'id':'clip'})
    assert selected==path


def test_explicit_variant_found_even_when_public_scene_url_missing(isolated_store):
    path,_=fixture_scene(isolated_store,['trajectory'])
    selected,_=routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',output_modalities=['trajectory']),'clip',{'id':'clip'})
    assert selected==path


def test_explicit_missing_outputs_never_fall_back_to_another_variant(isolated_store):
    path,_=fixture_scene(isolated_store,['trajectory']);fixture_scene(isolated_store,legacy=True)
    with pytest.raises(HTTPException) as exc:
        routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',output_modalities=['trajectory','contact']),'clip',{'scene_url':'/assets/'+path.relative_to(isolated_store).as_posix()})
    assert exc.value.detail['code']=='OUTPUT_VARIANT_UNAVAILABLE'


def test_legacy_fallback_requires_matching_scale_robot_and_no_selected_variant(isolated_store):
    path,_=fixture_scene(isolated_store,legacy=True,public=True)
    public={'scene_url':'/assets/legacy/scene.json'}
    selected,_=routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',trajectory_scale='preserve'),'clip',public)
    assert selected==path
    for body in (routes.ExportRequest(scene_id='clip',robot_id='g1',trajectory_scale='match_robot'),routes.ExportRequest(scene_id='clip',robot_id='x2',trajectory_scale='preserve')):
        with pytest.raises(HTTPException):routes.select_export_scene(body,'clip',public)
    current,_=fixture_scene(isolated_store,['trajectory','point_cloud'])
    selected,_=routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',trajectory_scale='preserve'),'clip',public)
    assert selected==current


@pytest.mark.parametrize('selectors', [{}, {'trajectory_scale':'preserve'}])
def test_complete_source_cannot_export_stale_public_result(isolated_store, monkeypatch, selectors):
    from arenalab_workbench import source_identity
    # The original result remains a valid historical artifact. Replacing the
    # registered source cannot make it the new recording's current result.
    path,_=fixture_scene(isolated_store,legacy=True,public=True)
    before=path.read_bytes()
    monkeypatch.setattr(source_identity,'registered_source',lambda *a,**k: ({}, {'format':'robot_joint_state'}))
    monkeypatch.setattr('arenalab_workbench.unified_scene.scene_key',lambda *a,**k:'current-native-source')
    with pytest.raises(HTTPException) as exc:
        routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',**selectors),
                                   'clip',{'scene_url':'/assets/legacy/scene.json'})
    assert exc.value.detail['code']=='SOURCE_REVISION_MISMATCH'
    assert path.read_bytes()==before


def test_default_complete_source_export_accepts_current_scaled_variant(isolated_store, monkeypatch):
    from arenalab_workbench import source_identity
    path,_=fixture_scene(isolated_store,['trajectory'],mode='match_robot')
    key=path.parent.name
    monkeypatch.setattr(source_identity,'registered_source',lambda *a,**k: ({}, {'format':'robot_joint_state'}))
    checked=[]
    def current_key(scene_id,mode,robot_id,outputs):
        checked.append((scene_id,mode,robot_id,outputs));return key
    monkeypatch.setattr('arenalab_workbench.unified_scene.scene_key',current_key)
    result,_=routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1'),
                                       'clip',{'scene_url':'/assets/'+path.relative_to(isolated_store).as_posix()})
    assert result==path and checked==[('clip','match_robot','g1',['trajectory'])]


def historical_source_binding(root):
    """A real saved input receipt from an earlier decoder version, without FK."""
    source=root/'original.npz'
    times=np.arange(3)/30
    np.savez(source,poses=np.zeros((3,165)),timestamps_s=times)
    objects=root/'objects.npz';np.savez(objects,timestamps_s=times,position=np.zeros((3,3)))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    spec=dict(path=str(source),format='smpl_parameters',family='smplx',world_up='z',units='m',
              floor_z_m=0.,objects_path=str(objects),sha256=digest(source))
    row=dict(source=str(source),frames=3,fps=30.)
    work=root/'old-source-work';work.mkdir()
    saved=dict(source_sha256=digest(source),source_spec={k:v for k,v in spec.items() if k!='path'},frames=3,
               decoder_version='old-code-version')
    (work/'input_manifest.json').write_text(json.dumps(saved))
    np.savez(work/'native_source.npz',timestamps_s=times,source_frame_indices=np.arange(3))
    (work/'objects.npz').write_bytes(objects.read_bytes())
    robot_folder=root/'historical-robot';robot_folder.mkdir()
    validation=dict(source_work=str(work),source_hashes={name:digest(work/name) for name in
                    ('input_manifest.json','native_source.npz','objects.npz')},motion_backend_fingerprint='old-refiner-code')
    (robot_folder/'validation.json').write_text(json.dumps(validation))
    return spec,row,work,validation


@pytest.mark.parametrize('selectors',[{}, {'trajectory_scale':'preserve','output_modalities':['trajectory']}])
@pytest.mark.parametrize('change',[None,'state','descriptor','objects','clock','receipt'])
def test_previous_code_result_exports_only_when_its_real_source_still_matches(isolated_store,monkeypatch,change,selectors):
    from arenalab_workbench import source_identity
    spec,row,work,validation=historical_source_binding(isolated_store)
    path,scene=fixture_scene(isolated_store,['trajectory'],public=True)
    scene['robots'][0]['playback_url']='/assets/historical-robot/playback.json'
    path.write_text(json.dumps(scene))
    if change=='state':
        np.savez(spec['path'],poses=np.ones((3,165)),timestamps_s=np.arange(3)/30)
    elif change=='descriptor':spec.update(format='robot_joint_state',family='robot',robot_id='g1')
    elif change=='objects':np.savez(spec['objects_path'],timestamps_s=np.arange(3)/30,position=np.ones((3,3)))
    elif change=='clock':
        np.savez(work/'native_source.npz',timestamps_s=1+np.arange(3)/30,source_frame_indices=np.arange(3))
        validation['source_hashes']['native_source.npz']=hashlib.sha256((work/'native_source.npz').read_bytes()).hexdigest()
        (isolated_store/'historical-robot/validation.json').write_text(json.dumps(validation))
    elif change=='receipt':(work/'input_manifest.json').write_text('{}')
    monkeypatch.setattr(source_identity,'registered_source',lambda *a,**k:(row,spec))
    monkeypatch.setattr('arenalab_workbench.unified_scene.scene_key',lambda *a,**k:'new-decoder-scene-key')
    request=routes.ExportRequest(scene_id='clip',robot_id='g1',**selectors)
    if change is None:
        result,_=routes.select_export_scene(request,'clip',{'scene_url':'/assets/legacy/scene.json'})
        assert result==path # A release/code hash change alone is not a source change.
    else:
        with pytest.raises(HTTPException) as exc:
            routes.select_export_scene(request,'clip',{'scene_url':'/assets/legacy/scene.json'})
        assert exc.value.detail['code']=='SOURCE_REVISION_MISMATCH'


@pytest.mark.parametrize('outputs',[['point_cloud'],['contact'],['inpaint_fov']])
def test_scene_only_and_internal_fov_trajectory_cannot_export_robot_dataset(isolated_store,outputs):
    path,_=fixture_scene(isolated_store,outputs)
    for requested in (outputs,None):
        with pytest.raises(HTTPException) as exc:
            routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',output_modalities=requested),'clip',{'scene_url':'/assets/'+path.relative_to(isolated_store).as_posix()})
        assert exc.value.detail['code']=='OUTPUT_TRAJECTORY_NOT_SELECTED'


def test_incomplete_or_mismatched_modern_variant_does_not_fall_back(isolated_store):
    path,scene=fixture_scene(isolated_store,['trajectory','point_cloud']);fixture_scene(isolated_store,legacy=True,public=True)
    scene['output_status']['point_cloud']='failed';path.write_text(json.dumps(scene))
    with pytest.raises(HTTPException) as exc:
        routes.select_export_scene(routes.ExportRequest(scene_id='clip',robot_id='g1',trajectory_scale='preserve'),'clip',{'scene_url':'/assets/legacy/scene.json'})
    assert exc.value.detail['code']=='OUTPUTS_INCOMPLETE'


@pytest.mark.parametrize('outputs',[[],['trajectory','trajectory']])
def test_direct_export_endpoint_rejects_empty_duplicate_selectors(outputs):
    with pytest.raises(ValidationError):routes.ExportRequest(scene_id='clip',robot_id='g1',output_modalities=outputs)


@pytest.mark.parametrize('include_cloud', [False, True])
def test_export_builds_real_archive_without_video_or_artifact_revision_copies(isolated_store,tmp_path,include_cloud):
    pytest.importorskip('pyarrow');trimesh=pytest.importorskip('trimesh');mj=pytest.importorskip('mujoco')
    folder=isolated_store/'robot';folder.mkdir()
    mesh=trimesh.Scene()
    for index in range(2):mesh.add_geometry(trimesh.creation.box(),node_name=f'geom_{index}',geom_name=f'geom_{index}')
    (folder/'model.glb').write_bytes(mesh.export(file_type='glb'))
    motion={'fps':30,'frames':3,'joints':{'names':['joint'],'values':[[0.],[.1],[.2]],'unit':'rad'},'positions':[[[0,0,0]]]*3,'quaternions':[[[0,0,0,1]]]*3}
    (folder/'playback.json').write_text(json.dumps(motion))
    native=tmp_path/'native';native.mkdir()
    model_path=tmp_path/'local-model.xml'
    model_path.write_text('<mujoco><worldbody><body name="base"><freejoint name="root"/><geom size=".1"/>'
                          '<body name="arm"><joint name="joint"/><geom size=".1"/></body></body></worldbody></mujoco>')
    model=mj.MjModel.from_xml_path(str(model_path));qpos=np.tile(model.qpos0,(3,1));qpos[:,7]=[0,.1,.2]
    np.savez(native/'motion.npz',qpos=qpos,fps=30)
    from scipy.spatial.transform import Rotation
    basis=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    state=mj.MjData(model);positions=[];rotations=[]
    for row in qpos:
        state.qpos[:]=row;mj.mj_forward(model,state)
        positions.append(state.geom_xpos.copy());rotations.append(state.geom_xmat.copy().reshape(-1,3,3))
    positions,rotations=np.asarray(positions),np.asarray(rotations)
    np.savez(native/'robot-render.npz',positions=positions,rotations=rotations,native_body_geom_ids=np.arange(model.ngeom))
    motion['positions']=(positions@basis.T).tolist()
    motion['quaternions']=Rotation.from_matrix((basis@rotations).reshape(-1,3,3)).as_quat().reshape(3,model.ngeom,4).tolist()
    (folder/'playback.json').write_text(json.dumps(motion))
    (native/'request.json').write_text(json.dumps(dict(robot_id='g1',output_dir=str(native),row={'xml':str(model_path)})))
    (folder/'validation.json').write_text(json.dumps(dict(native_output_dir=str(native))))
    scene={'id':'clip','title':'Motion only','dataset':'Synthetic','fps':30,'frames':3,'video':None,'robots':[{'id':'g1','model_url':'/assets/robot/model.glb','playback_url':'/assets/robot/playback.json'}],'output_modalities':['trajectory'],'completed_outputs':['trajectory'],
        'output_status':{'trajectory':'completed'},'generation_status':{'status':'completed'},
        'cloud':{'url':'/assets/unselected-missing-cloud.bin'},'patch_videos':{'g1':'/assets/unselected-missing-fov.mp4'},'tactile':{'url':'/assets/unselected-missing-contact.json'},'objects':[]}
    scene['source_world_to_display']=basis.tolist()
    if include_cloud:
        cloud=isolated_store/'geometry';cloud.mkdir();(cloud/'index.json').write_text('{}')
        revisions=cloud/'.artifact-revisions/previous';revisions.mkdir(parents=True)
        (revisions/'index.json').write_text('historical download revision')
        scene.update(output_modalities=['trajectory','point_cloud'],completed_outputs=['trajectory','point_cloud'],
                     output_status={'trajectory':'completed','point_cloud':'completed'},
                     cloud={'format':'depth-rgb-v1','geometry_url':'/assets/geometry','frames':[{'source_frame':0},{'source_frame':2}]})
    source=isolated_store/'scene.json';source.write_text(json.dumps(scene))
    perception=tmp_path/'perception';perception.mkdir();np.savez(perception/'depth.npz',depth=np.ones((2,2)))
    result=exporter.export_dataset(source,'g1',tmp_path/'dataset',raw_sources=[('perception',perception)],video_sources={})
    assert Path(result['path']).is_file()
    info=json.loads((tmp_path/'dataset/meta/info.json').read_text());assert info['total_videos']==0
    extension=tmp_path/'dataset/extensions/arenalabs';saved=json.loads((extension/'scene.json').read_text())
    assert saved['video'] is None and not saved.get('patch_videos')
    if include_cloud:
        cloud=tmp_path/'dataset'/saved['cloud']['geometry_url']
        assert (cloud/'index.json').read_text()=='{}'
        assert not list(cloud.rglob('.artifact-revisions'))
    else:
        assert 'cloud' not in saved and not (extension/'raw/perception').exists()
    assert (extension/'replay/robot.usda').is_file()


@pytest.mark.parametrize('failure', ['pending','failed','missing_status','missing_completed',
                                    'running','legacy','duplicate','unknown'])
def test_new_export_cannot_promote_incomplete_or_legacy_scene_to_ready(tmp_path,failure):
    scene={'id':'clip','robots':[{'id':'g1'}],
           'output_modalities':['trajectory','contact'],
           'completed_outputs':['trajectory','contact'],
           'output_status':{'trajectory':'completed','contact':'completed'},
           'generation_status':{'status':'completed'}}
    if failure in {'pending','failed'}:scene['output_status']['contact']=failure
    elif failure=='missing_status':scene.pop('output_status')
    elif failure=='missing_completed':scene['completed_outputs']=['trajectory']
    elif failure=='running':scene['generation_status']['status']='running'
    elif failure=='legacy':
        scene['generation_status']['output_modalities']=scene.pop('output_modalities')
    elif failure=='duplicate':scene['output_modalities'].append('contact')
    else:scene['output_modalities'].append('unknown')
    source=tmp_path/'scene.json';source.write_text(json.dumps(scene));before=source.read_bytes()
    # Existing published archives are unaffected by the stricter new-build gate.
    existing=tmp_path/'existing.zip';existing.write_bytes(b'immutable older archive')
    with pytest.raises(ValueError,match='OUTPUTS_INCOMPLETE|EXPORT_CURRENT_FORMAT_REQUIRED'):
        exporter.export_dataset(source,'g1',tmp_path/'new-dataset')
    assert source.read_bytes()==before and existing.read_bytes()==b'immutable older archive'
    assert not (tmp_path/'new-dataset').exists() and not (tmp_path/'new-dataset.zip').exists()


@pytest.mark.parametrize('existing_kind',['folder','archive','partial','dangling_archive_symlink'])
def test_export_never_replaces_existing_folder_archive_or_partial(tmp_path,existing_kind):
    scene={'id':'clip','robots':[{'id':'g1'}],'output_modalities':['trajectory'],
           'completed_outputs':['trajectory'],'output_status':{'trajectory':'completed'},
           'generation_status':{'status':'completed'}}
    source=tmp_path/'scene.json';source.write_text(json.dumps(scene))
    output=tmp_path/'dataset';archive=output.with_suffix('.zip');pending=archive.with_suffix('.zip.partial')
    if existing_kind=='folder':
        output.mkdir();sentinel=output/'untouched';sentinel.write_bytes(b'keep')
    elif existing_kind=='dangling_archive_symlink':
        archive.symlink_to(tmp_path/'missing-original');sentinel=None
    else:
        sentinel=archive if existing_kind=='archive' else pending;sentinel.write_bytes(b'keep')
    with pytest.raises(FileExistsError,match='immutable'):
        exporter.export_dataset(source,'g1',output)
    if sentinel is not None:assert sentinel.read_bytes()==b'keep'
    else:assert archive.is_symlink()
    if existing_kind!='folder':assert not output.exists()


def test_collect_motion_only_video_sources_is_empty():
    with patch('arenalab_workbench.gripper_registry.sources',return_value={}),patch('arenalab_workbench.scene_job.resolve_clip',return_value=({},{'video':None})):
        assert exporter.collect_video_sources('clip',{})=={}
