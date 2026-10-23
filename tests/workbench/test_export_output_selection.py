"""Export selects immutable deliverables, including video-free robot datasets."""
import json
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


def test_motion_only_export_builds_real_archive_without_video_or_unselected_dependencies(isolated_store,tmp_path):
    pytest.importorskip('pyarrow');trimesh=pytest.importorskip('trimesh')
    folder=isolated_store/'robot';folder.mkdir()
    mesh=trimesh.Scene();mesh.add_geometry(trimesh.creation.box(),node_name='geom_0',geom_name='geom_0')
    (folder/'model.glb').write_bytes(mesh.export(file_type='glb'))
    motion={'fps':30,'frames':3,'joints':{'names':['joint'],'values':[[0.],[.1],[.2]],'unit':'rad'},'positions':[[[0,0,0]]]*3,'quaternions':[[[0,0,0,1]]]*3}
    (folder/'playback.json').write_text(json.dumps(motion))
    scene={'id':'clip','title':'Motion only','dataset':'Synthetic','fps':30,'frames':3,'video':None,'robots':[{'id':'g1','model_url':'/assets/robot/model.glb','playback_url':'/assets/robot/playback.json'}],'output_modalities':['trajectory'],'completed_outputs':['trajectory'],
        'cloud':{'url':'/assets/unselected-missing-cloud.bin'},'patch_videos':{'g1':'/assets/unselected-missing-fov.mp4'},'tactile':{'url':'/assets/unselected-missing-contact.json'},'objects':[]}
    source=isolated_store/'scene.json';source.write_text(json.dumps(scene))
    perception=tmp_path/'perception';perception.mkdir();np.savez(perception/'depth.npz',depth=np.ones((2,2)))
    result=exporter.export_dataset(source,'g1',tmp_path/'dataset',raw_sources=[('perception',perception)],video_sources={})
    assert Path(result['path']).is_file()
    info=json.loads((tmp_path/'dataset/meta/info.json').read_text());assert info['total_videos']==0
    extension=tmp_path/'dataset/extensions/arenalabs';saved=json.loads((extension/'scene.json').read_text())
    assert saved['video'] is None and 'cloud' not in saved and not saved.get('patch_videos')
    assert not (extension/'raw/perception').exists()
    assert (extension/'replay/robot.usda').is_file()


def test_collect_motion_only_video_sources_is_empty():
    with patch('arenalab_workbench.gripper_registry.sources',return_value={}),patch('arenalab_workbench.scene_job.resolve_clip',return_value=({},{'video':None})):
        assert exporter.collect_video_sources('clip',{})=={}
