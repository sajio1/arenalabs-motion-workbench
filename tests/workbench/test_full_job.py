"""Scene completion cannot masquerade as completion of all robot modalities."""
import json
from arenalab_workbench.progressive_job import Progress
from arenalab_workbench.job_status import job_view


def test_progressive_modalities_finish_independently(tmp_path):
    status = tmp_path / 'job' / 'state.json'
    status.parent.mkdir()
    progress = Progress(status, '/assets/scene.json', 'robot')
    progress.publish('motion', 'Motion ready')
    state = json.loads(status.read_text())
    assert state['status'] == 'running'
    assert state['robot_ready'] is True
    progress.publish('scene', 'Scene ready')
    assert json.loads(status.read_text())['status'] == 'running'
    progress.publish('fov', 'FOV ready')
    progress.publish('contact', 'Contact ready')
    progress.complete()
    view = job_view({'id': 'job', 'created_at': 0}, {}, tmp_path)
    assert view['status'] == 'completed'
    assert view['scene_url'] == '/assets/scene.json'


from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
from arenalab_workbench.output_modalities import OUTPUTS


def _harness(monkeypatch, tmp_path, outputs, scale='preserve'):
    from arenalab_workbench import full_job, gripper_job, robot_registry, source_requirements, body_prior, gpu_scheduler
    store = tmp_path/'store'; store.mkdir()
    source = tmp_path/'raw.h5'; source.write_bytes(b'original observations')
    body = tmp_path/'body'
    calls=[]
    public=dict(id='clip',title='Test',dataset='EgoDex',end_effector='wuji')
    row=dict(source=str(source),frames=8,fps=30,video=dict(path=str(tmp_path/'video.mp4')))
    monkeypatch.setenv('ARENA_OUTPUT_MODALITIES',json.dumps(outputs))
    monkeypatch.setenv('ARENA_TRAJECTORY_SCALE',scale)
    monkeypatch.setattr(source_requirements,'input_requirements',lambda *args:dict(output_capabilities=dict(outputs={name:dict(available=True) for name in OUTPUTS})))
    monkeypatch.setattr(full_job,'STORE',store)
    monkeypatch.setattr(full_job,'ROOT',tmp_path/'worker')
    monkeypatch.setattr(full_job.paths,'store',store)
    monkeypatch.setattr(gpu_scheduler,'ROOT',tmp_path/'scheduler')
    monkeypatch.setattr(full_job,'resolve_clip',lambda _: (public,row))
    monkeypatch.setattr(gripper_job,'sources',lambda: {})
    monkeypatch.setattr(robot_registry,'roster',lambda: {'r':dict(id='r',kind='Humanoid',pipeline_row={})})
    def folder(_):
        calls.append('body_folder');return body
    monkeypatch.setattr(full_job,'body_folder',folder)
    monkeypatch.setattr(full_job,'depth_current',lambda scene:scene.get('geometry',{}).get('ready') is True)
    monkeypatch.setattr(full_job,'native_cache_valid',lambda *args,**kw:False)
    def scene_run(*args):
        calls.append('scene')
        path=store/'fresh/scene.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(cloud=dict(url='/assets/cloud.npz'),objects=[dict(id='box')],geometry=dict(ready=True,scene_work='cache'))))
        return '/assets/fresh/scene.json'
    monkeypatch.setattr(full_job,'scene_run',scene_run)
    def prepare(*args):
        calls.append('prepare_observations');body.mkdir(exist_ok=True)
        (body/'input_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
        (body/'source_hands.npz').write_bytes(b'observed hands')
        return body
    monkeypatch.setattr(full_job,'prepare_contact_source',prepare)
    def subprocess_run(command,env=None,**kwargs):
        script=Path(command[1]).name;calls.append(script)
        if script=='body_prior.py':
            body.mkdir(exist_ok=True);(body/'prior-ready.json').write_text('{}')
        elif script=='retarget_robot.py':
            dest=Path(env['ARENA_EXPORT_DIR'])
            gpu_scheduler.merge_scene(dest/'scene.json',lambda scene:scene.update(robots=[dict(id='r',hand='wuji',playback_url='playback')]))
            robot=dest/'robot-r';robot.mkdir(exist_ok=True)
            (robot/'playback.json').write_text('{}');(robot/'model.glb').write_bytes(b'mesh')
        elif script=='render_recording.py':
            dest=Path(env['ARENA_EXPORT_DIR'])
            gpu_scheduler.merge_scene(dest/'scene.json',lambda scene:scene.update(patch_videos={'r':dict(url='/assets/fov.mp4')}))
        elif script=='contact_recording.py':
            assert env['ARENA_DEFER_PROXIMITY']=='1'
            dest=Path(env['ARENA_EXPORT_DIR'])
            data=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_hands_sha256=hashlib.sha256((body/'source_hands.npz').read_bytes()).hexdigest())
            (dest/'video-contact.json').write_text(json.dumps(data))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(full_job,'gpu_run',subprocess_run)
    monkeypatch.setattr(full_job.subprocess,'run',subprocess_run)
    from arenalab_workbench import depth_store
    monkeypatch.setattr(depth_store,'fov_current',lambda scene,rid:rid in scene.get('patch_videos',{}))
    def scale_scene(*args,**kwargs):calls.append('fit_scale')
    monkeypatch.setattr(full_job,'apply_scene_scale',scale_scene)
    status=tmp_path/'job/state.json'
    def run():
        url=full_job._run('clip','r','wuji',status)
        path=store/url.removeprefix('/assets/')
        return json.loads(path.read_text()),json.loads(status.read_text()),path.parent
    return run,calls,full_job


def test_cloud_only_never_prepares_or_infers_body(monkeypatch,tmp_path):
    run,calls,_=_harness(monkeypatch,tmp_path,['point_cloud'])
    scene,state,_=run()
    assert calls==['scene']
    assert scene['robots']==[] and scene['cloud']
    assert state['completed_outputs']==['point_cloud']
    assert state['scene_ready'] and not state['robot_ready']
    assert state['phase']=='Selected outputs ready'


def test_trajectory_only_skips_cv_contact_and_render(monkeypatch,tmp_path):
    run,calls,_=_harness(monkeypatch,tmp_path,['trajectory'])
    scene,state,_=run()
    assert set(calls)=={'body_folder','body_prior.py','retarget_robot.py'}
    assert scene['robots'] and 'cloud' not in scene and 'patch_videos' not in scene
    assert state['completed_outputs']==['trajectory'] and not state['scene_ready']


def test_contact_only_prepares_observations_without_body_or_geometry(monkeypatch,tmp_path):
    run,calls,_=_harness(monkeypatch,tmp_path,['contact'])
    scene,state,_=run()
    assert set(calls)=={'prepare_observations','contact_recording.py'}
    assert scene['robots']==[] and 'cloud' not in scene
    assert scene['tactile']['status']=='video_estimate'
    assert state['contact_ready'] and not state['robot_ready'] and not state['scene_ready']
    calls.clear();run()
    assert calls==['prepare_observations']  # Reuse source-hash-verified video contact.


def test_fov_dependencies_remain_internal_and_are_reusable(monkeypatch,tmp_path):
    run,calls,module=_harness(monkeypatch,tmp_path,['inpaint_fov'])
    scene,state,dest=run()
    assert set(calls)=={'body_folder','body_prior.py','retarget_robot.py','scene','render_recording.py'}
    assert scene['robots']==[] and 'cloud' not in scene and scene['patch_videos']
    assert state['completed_outputs']==['inpaint_fov']
    assert state['fov_ready'] and not state['robot_ready'] and not state['scene_ready']
    from arenalab_workbench.unified_scene import scene_key
    worker=module.STORE/'manipulation4d/unified'/scene_key('clip','preserve','r')/'artifacts'
    internal=json.loads((worker/'scene.json').read_text())
    assert internal['robots'] and internal['cloud']
    monkeypatch.setattr(module,'native_cache_valid',lambda *args,**kw:True)
    calls.clear();run()
    assert set(calls)=={'body_folder','body_prior.py'}


def test_cloud_match_scale_only_runs_body_and_cpu_scale_dependency(monkeypatch,tmp_path):
    run,calls,_=_harness(monkeypatch,tmp_path,['point_cloud'],'match_robot')
    scene,state,_=run()
    assert set(calls)=={'body_folder','body_prior.py','scene','fit_scale'}
    assert scene['robots']==[] and state['dependency_plan']['source_body_for_scene_scale']


def test_source_stage_failure_is_explicit_and_not_completed(monkeypatch,tmp_path):
    run,calls,module=_harness(monkeypatch,tmp_path,['point_cloud'])
    def fail(*args):raise RuntimeError('Depth inference unavailable')
    monkeypatch.setattr(module,'scene_run',fail)
    with pytest.raises(RuntimeError,match='Depth inference unavailable'):run()
    state=json.loads((tmp_path/'job/state.json').read_text())
    assert state['status']=='failed' and state['output_status']=={'point_cloud':'failed'}
    assert state['completed_outputs']==[]


def test_selection_unavailable_does_not_start_workers(monkeypatch,tmp_path):
    run,calls,_=_harness(monkeypatch,tmp_path,['contact'])
    from arenalab_workbench import source_requirements
    monkeypatch.setattr(source_requirements,'input_requirements',lambda *args:dict(output_capabilities=dict(outputs={'trajectory':dict(available=True),'contact':dict(available=False,reason='No observed hands')})))
    with pytest.raises(ValueError,match='No observed hands'):run()
    assert calls==[]


def test_new_output_selection_reuses_shared_motion_and_scene(monkeypatch,tmp_path):
    run,calls,module=_harness(monkeypatch,tmp_path,['trajectory'])
    first,_,_=run()
    monkeypatch.setattr(module,'native_cache_valid',lambda *args,**kw:True)
    monkeypatch.setenv('ARENA_OUTPUT_MODALITIES',json.dumps(['trajectory','point_cloud']))
    calls.clear();scene,state,_=run()
    assert 'retarget_robot.py' not in calls
    assert 'scene' in calls and scene['robots'] and scene['cloud']
    assert first['robots']==scene['robots']
    assert state['completed_outputs']==['trajectory','point_cloud']


def test_fov_unavailable_receipt_is_not_claimed_as_completed(monkeypatch,tmp_path):
    run,calls,module=_harness(monkeypatch,tmp_path,['inpaint_fov'])
    original=module.gpu_run
    def worker(command,**kw):
        result=original(command,**kw)
        if Path(command[1]).name=='render_recording.py':
            path=Path(kw['env']['ARENA_EXPORT_DIR'])/'scene.json';scene=json.loads(path.read_text())
            scene['patch_unavailable']={'r':'No supported source-camera compositor'}
            path.write_text(json.dumps(scene))
        return result
    monkeypatch.setattr(module,'gpu_run',worker)
    with pytest.raises(ValueError,match='No supported source-camera compositor'):run()
    state=json.loads((tmp_path/'job/state.json').read_text())
    assert state['output_status']=={'inpaint_fov':'failed'} and not state['fov_ready']


def _gripper_harness(monkeypatch,tmp_path,outputs):
    from arenalab_workbench import gripper_job, gripper_body, robot_registry, source_requirements, gpu_scheduler, height_registry
    store=tmp_path/'store';store.mkdir()
    work=tmp_path/'source';work.mkdir();(work/'source.npz').write_bytes(b'observations')
    (work/'source.json').write_text(json.dumps(dict(dataset='Test robot',frames=8)))
    public=dict(id='clip',title='Recorded grippers',dataset='Test robot',end_effector='robotiq-2f-140')
    calls=[];body=work/'body';status=tmp_path/'job/state.json'
    monkeypatch.setenv('ARENA_OUTPUT_MODALITIES',json.dumps(outputs));monkeypatch.setenv('ARENA_TRAJECTORY_SCALE','preserve')
    monkeypatch.setattr(source_requirements,'input_requirements',lambda *args:dict(output_capabilities=dict(outputs={name:dict(available=name!='contact') for name in OUTPUTS})))
    monkeypatch.setattr(gripper_job.paths,'store',store)
    monkeypatch.setattr(gpu_scheduler,'ROOT',tmp_path/'scheduler')
    monkeypatch.setattr(gripper_job,'sources',lambda:dict(clip=dict(work=str(work),destination=str(tmp_path/'old'))))
    monkeypatch.setattr(gripper_job,'depth_current',lambda scene:scene.get('geometry',{}).get('ready') is True)
    monkeypatch.setattr(gripper_job,'cached',lambda *args,**kw:False)
    monkeypatch.setattr(gripper_job,'fov_current',lambda scene,rid:rid in scene.get('patch_videos',{}))
    monkeypatch.setattr(gripper_body,'body_path',lambda _:body)
    monkeypatch.setattr(height_registry,'public_entry',lambda _:public)
    monkeypatch.setattr(robot_registry,'roster',lambda: {'r':dict(id='r',kind='Humanoid',pipeline_row={})})
    def worker(command,**kw):
        script=Path(command[1]).name;calls.append(script)
        if script=='gripper_body.py':
            body.mkdir(exist_ok=True);(body/'prior-ready.json').write_text('{}')
        elif script=='gripper_scene.py':
            dest=Path(command[3]);dest.mkdir(parents=True,exist_ok=True)
            (dest/'scene.json').write_text(json.dumps(dict(cloud=dict(url='/assets/cloud'),geometry=dict(ready=True,source_sha256=hashlib.sha256((work/'source.npz').read_bytes()).hexdigest(),visual_policy='gripper-visual-v4'))))
        elif script=='gripper_export.py':
            dest=Path(command[4])
            gpu_scheduler.merge_scene(dest/'scene.json',lambda scene:scene.update(robots=[dict(id='r',hand='robotiq-2f-140')]))
            robot=dest/'robot-r';robot.mkdir(exist_ok=True)
            (robot/'playback.json').write_text('{}')
        elif script=='gripper_render.py':
            dest=Path(command[4])
            gpu_scheduler.merge_scene(dest/'scene.json',lambda scene:scene.update(patch_videos={'r':'/assets/fov.mp4'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(gripper_job,'gpu_run',worker);monkeypatch.setattr(gripper_job.subprocess,'run',worker)
    def run():
        url=gripper_job.run('clip','r','robotiq-2f-140',status)
        return json.loads((store/url.removeprefix('/assets/')).read_text()),json.loads(status.read_text())
    return run,calls,gripper_job


def test_gripper_cloud_only_has_no_body_or_trajectory_dependencies(monkeypatch,tmp_path):
    run,calls,_=_gripper_harness(monkeypatch,tmp_path,['point_cloud'])
    scene,state=run()
    assert calls==['gripper_scene.py'] and scene['robots']==[]
    assert state['completed_outputs']==['point_cloud']


def test_gripper_trajectory_only_has_no_cv_or_fov(monkeypatch,tmp_path):
    run,calls,_=_gripper_harness(monkeypatch,tmp_path,['trajectory'])
    scene,state=run()
    assert set(calls)=={'gripper_body.py','gripper_retarget.py','gripper_export.py'}
    assert scene['robots'] and 'cloud' not in scene and 'patch_videos' not in scene
    assert state['completed_outputs']==['trajectory']


def test_gripper_fov_filters_internal_artifacts_and_reuses_cache(monkeypatch,tmp_path):
    run,calls,module=_gripper_harness(monkeypatch,tmp_path,['inpaint_fov'])
    scene,state=run()
    assert set(calls)=={'gripper_body.py','gripper_retarget.py','gripper_export.py','gripper_scene.py','gripper_render.py'}
    assert scene['robots']==[] and 'cloud' not in scene and scene['patch_videos']
    assert state['completed_outputs']==['inpaint_fov']
    monkeypatch.setattr(module,'cached',lambda *args,**kw:True)
    calls.clear();run()
    assert calls==['gripper_body.py']


def test_gripper_contact_unsupported_does_not_execute_workers(monkeypatch,tmp_path):
    run,calls,_=_gripper_harness(monkeypatch,tmp_path,['contact'])
    with pytest.raises(ValueError,match='OUTPUTS_UNAVAILABLE'):run()
    assert calls==[]


def test_contact_observation_staging_preserves_raw_points_and_is_immutable(monkeypatch,tmp_path):
    import h5py
    import numpy as np
    from arenalab_workbench.job_outputs import prepare_contact_source
    from arenalab_workbench import gpu_scheduler
    monkeypatch.setattr(gpu_scheduler,'ROOT',tmp_path/'locks')
    source=tmp_path/'recording.h5';video=tmp_path/'video.mp4';video.write_bytes(b'raw RGB')
    with h5py.File(source,'w') as data:
        for side in ('left','right'):
            names=[side+'Hand']+[side+'Thumb'+part for part in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
            names += [side+finger+'Finger'+part for finger in ('Index','Middle','Ring','Little') for part in ('Knuckle','IntermediateBase','IntermediateTip','Tip')]
            for i,name in enumerate(names):
                transform=np.tile(np.eye(4),(5,1,1));transform[:,:3,3]=np.arange(15).reshape(5,3)+i
                data['transforms/'+name]=transform;data['confidences/'+name]=np.linspace(.1,1,5)
    public=dict(id='clip',dataset='EgoDex')
    row=dict(source=str(source),frames=3,source_window=dict(start_frame=1),video=dict(path=str(video)))
    path=prepare_contact_source(public,row,tmp_path/'contact')
    with np.load(path/'source_hands.npz') as prepared:
        restored=prepared['points']@prepared['source_to_zup']
        np.testing.assert_array_equal(restored[:,0,0],np.arange(15).reshape(5,3)[1:4])
        np.testing.assert_array_equal(prepared['confidence'][:,0,0],np.linspace(.1,1,5)[1:4])
    manifest=path/'input_manifest.json';before=manifest.read_bytes()
    assert prepare_contact_source(public,row,tmp_path/'contact')==path and manifest.read_bytes()==before
    assert not (path/'model.npz').exists() and not (path/'prior-ready.json').exists()
    manifest.write_text('{}')
    with pytest.raises(ValueError,match='Immutable contact observations changed'):prepare_contact_source(public,row,tmp_path/'contact')


def test_missing_point_cloud_artifact_fails_instead_of_placeholder_success(monkeypatch,tmp_path):
    run,calls,module=_harness(monkeypatch,tmp_path,['point_cloud'])
    def unavailable(*args):
        path=module.STORE/'fresh/scene.json';path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(geometry=dict(status='unavailable',reason='No RGB input'))))
        return '/assets/fresh/scene.json'
    monkeypatch.setattr(module,'scene_run',unavailable)
    with pytest.raises(ValueError,match='No RGB input'):run()
    state=json.loads((tmp_path/'job/state.json').read_text())
    scene=json.loads((module.STORE/state['scene_url'].removeprefix('/assets/')).read_text())
    assert state['status']=='failed' and not state['scene_ready']
    assert scene['generation_status']['status']=='failed'
    assert scene['output_status']=={'point_cloud':'failed'}


def test_native_cache_rejects_changed_supplied_body_attachment(monkeypatch,tmp_path):
    from arenalab_workbench.job_outputs import native_cache_valid
    from arenalab_workbench import motion_backend
    monkeypatch.setattr(motion_backend,'cache_matches',lambda report:True)
    source=tmp_path/'normalized';source.mkdir();(source/'native_source.npz').write_bytes(b'provided full body v1')
    monkeypatch.setattr(motion_backend,'normalized_source',lambda work:source)
    dest=tmp_path/'export';folder=dest/'robot-r';folder.mkdir(parents=True)
    output=tmp_path/'numerical';output.mkdir();prior=output/'native_prior.npz';prior.write_bytes(b'raw prior')
    for name in ('motion.npz','gpu_qpos.npz','report.json'):(output/name).write_bytes(b'complete')
    for name in ('model.glb','playback.json'):(folder/name).write_bytes(b'export')
    report=dict(native_output_dir=str(output),raw_sha256={str(prior):motion_backend.sha(prior)},source_sha256='camera-observations-sha')
    (folder/'validation.json').write_text(json.dumps(report))
    (dest/'scene.json').write_text(json.dumps(dict(robots=[dict(id='r')])))
    (output/'request.json').write_text(json.dumps(dict(source_sha256={'native_source.npz':motion_backend.sha(source/'native_source.npz')})))
    assert native_cache_valid(dest,'r',source_work=source,source_sha256='camera-observations-sha')
    (source/'native_source.npz').write_bytes(b'new provided body, same camera')
    assert not native_cache_valid(dest,'r',source_work=source,source_sha256='camera-observations-sha')


def test_unselected_motion_and_fov_annotations_are_filtered():
    from arenalab_workbench.job_outputs import filtered_scene
    scene=dict(robots=[dict(id='r')],source_body=dict(url='human'),hand_motion_audit_url='robot-hand-audit',
               cloud=dict(url='cloud'),patch_videos={'r':'video'},patch_kinds={'r':'native_preview'},patch_unavailable={'r':'old reason'},patch_depth_policy={'r':'old'})
    visible=filtered_scene(scene,dict(effective_outputs=['point_cloud']),dict(scene=True))
    assert visible['robots']==[] and visible['cloud']
    for key in ('source_body','hand_motion_audit_url','patch_videos','patch_kinds','patch_unavailable','patch_depth_policy'):
        assert key not in visible
    assert scene['robots'] and scene['hand_motion_audit_url']  # Internal cache is untouched.
