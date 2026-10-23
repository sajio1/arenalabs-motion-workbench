"""Named complete robot inputs bypass body completion and preserve source evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
mj=pytest.importorskip('mujoco')
pytest.importorskip('torch')
from arenalab_workbench import robot_motion_source as robot
from arenalab_workbench.motion_sources import source_route, prepare_body_source, validate_body_attachment
from arenalab_motion.sources import load_source
from arenalab_motion.scaling import materialize_source
from arenalab_motion.prior import encode_source
from test_motion_gpu_metadata import rig


def fixture(tmp_path,monkeypatch):
    row=rig(tmp_path)
    path=Path(row['xml']);path.write_text(path.read_text().replace('<worldbody>','<default><geom type="box" size=".02 .02 .02" group="1"/><joint limited="true" range="-160 160"/></default><worldbody>'))
    row['palm_frames']={s+'_wrist':dict(distal=[1,0,0],normal=[0,0,1]) for s in ('left','right')}
    from arenalab_motion.robot_metadata import ensure_robot_metadata
    row=ensure_robot_metadata(row)
    from arenalab_workbench import robot_registry
    monkeypatch.setattr(robot_registry,'roster',lambda *a,**kw:{'fixture':dict(pipeline_row=row)})
    model=mj.MjModel.from_xml_path(row['xml']);names=[model.joint(j).name for j in range(model.njnt) if model.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE]
    arrays=dict(joint_names=np.array(names[::-1]),joint_positions=np.arange(4*len(names)).reshape(4,-1)*.001,
                root_position=np.tile([.2,.1,1.],(4,1)),root_quaternion_wxyz=np.tile([1.,0,0,0],(4,1)),timestamps_s=np.arange(4)/30)
    source=tmp_path/'motion.npz';np.savez_compressed(source,**arrays)
    spec=dict(format='robot_joint_state',family='robot',robot_id='fixture',path=str(source),units='m',world_up='z',floor_z_m=0.,evidence='synthetic')
    return row,spec,arrays


def test_named_states_keep_every_joint_and_do_not_call_human_completion(tmp_path,monkeypatch):
    row,spec,raw=fixture(tmp_path,monkeypatch)
    from arenalab_workbench import body_inference
    monkeypatch.setattr(body_inference,'run',lambda *a,**k:pytest.fail('Robot source entered EgoAllo'))
    work=tmp_path/'work';work.mkdir();(work/'input_manifest.json').write_text(json.dumps(dict(motion_source=spec,clip='test',end_effector='none')))
    original=Path(spec['path']).read_bytes();out=prepare_body_source(work,tmp_path/'source')
    src=load_source(out)
    assert src['source_decoder_used'] is False and src['frames']==4
    assert src['names']==list(robot.read_state(spec)[0]['spec'].joint_names)
    assert not any(n in src['names'] for n in ['Hips','LeftHand'])
    assert source_route({'motion_source':spec})['route']=='supplied-robot'
    target,q,times,indices=robot.read_state(spec);model=target['model'];data=mj.MjData(model)
    with np.load(out/'native_source.npz') as z:
        for t in range(4):
            data.qpos[:]=q[t];mj.mj_forward(model,data)
            np.testing.assert_allclose(z['source_body_position_world_m'][t],data.xpos[target['keep']],atol=1e-12)
        np.testing.assert_allclose(z['timestamps_s'],raw['timestamps_s'],atol=0)
        assert z['dof'].shape[0]==len(src['names'])
    assert Path(spec['path']).read_bytes()==original
    import torch
    class Codec:
        device='cpu';_motion_model_identity={'fixture':'no-neural-quality-claim'}
        def encode(self,pose,skel,dof):
            assert pose.shape[1]==len(src['names'])
            return torch.zeros((len(pose),2,2),dtype=torch.long)
    encode_source(src,tmp_path/'prior',Codec())
    z=np.load(tmp_path/'prior/source_prime_original.npz')
    np.testing.assert_array_equal(z['observed_nodes'],np.arange(len(src['names'])))
    assert not bool(z['source_decoder_used'])
    report=json.loads((out/'input_manifest.json').read_text())
    assert report['human_model'] is None and not report['source_completion']
    assert not (out/'source_mesh.npz').exists()
    from arenalab_workbench import source_preview
    monkeypatch.setattr(source_preview.paths, 'store', tmp_path)
    preview = source_preview.export_source_preview(out, tmp_path/'assets')
    assert preview['source_family']=='robot' and preview['frames']==4
    from arenalab_workbench.source_display import source_scene_metadata, MANIPULATION_WORLD_TO_DISPLAY
    metadata = source_scene_metadata(preview, {}, MANIPULATION_WORLD_TO_DISPLAY)
    assert 'source_robot' in metadata and 'source_body' not in metadata
    with np.load(out/'source_robot_playback.npz') as visual:
        assert visual['positions'].shape[:2] == visual['rotations'].shape[:2]
        np.testing.assert_array_equal(visual['timestamps_s'], raw['timestamps_s'])


def test_robot_scaling_uses_robot_limbs_and_keeps_rotation_and_clock(tmp_path,monkeypatch):
    row,spec,_=fixture(tmp_path,monkeypatch)
    source=robot.prepare_robot_source(spec,tmp_path/'source')
    small=tmp_path/'smaller.xml';text=Path(row['xml']).read_text()
    # Uniform change to the target's neutral bone offsets; source stays intact.
    import re
    text=re.sub(r'pos="([^"]+)"',lambda m:'pos="'+' '.join(str(float(v)*.5) for v in m[1].split())+'"',text)
    small.write_text(text);target=dict(row,xml=str(small),name='small')
    report=materialize_source(source,target,tmp_path/'scaled')
    assert report['morphology_scale']['scale']==pytest.approx(.5)
    a=np.load(source/'native_source.npz');b=np.load(tmp_path/'scaled/native_source.npz')
    for key in ('timestamps_s','full_body_rotation_yup','reference_world_R_zup','dof'):
        np.testing.assert_array_equal(a[key],b[key])
    anchor=np.array(report['morphology_scale']['anchor_world_m'])
    np.testing.assert_allclose(b['semantic_position_world_m'],anchor+.5*(a['semantic_position_world_m']-anchor),atol=1e-12)
    assert load_source(tmp_path/'scaled')['frames']==4


@pytest.mark.parametrize('broken',['missing_joint','duplicate_joint','bad_root','nan_joint','bad_clock'])
def test_bad_robot_state_never_falls_back_to_three_points(tmp_path,monkeypatch,broken):
    _,spec,raw=fixture(tmp_path,monkeypatch)
    if broken=='missing_joint':raw['joint_names']=raw['joint_names'][:-1];raw['joint_positions']=raw['joint_positions'][:,:-1]
    if broken=='duplicate_joint':raw['joint_names'][0]=raw['joint_names'][1]
    if broken=='bad_root':raw['root_quaternion_wxyz'][0]*=2
    if broken=='nan_joint':raw['joint_positions'][1,0]=np.nan
    if broken=='bad_clock':raw['timestamps_s'][1]+=.01
    np.savez_compressed(spec['path'],**raw)
    with pytest.raises(ValueError):robot.read_state(spec)


def test_upload_rejects_model_paths_and_accepts_registered_robot(tmp_path,monkeypatch):
    _,spec,_=fixture(tmp_path,monkeypatch);spec['path']='motion.npz'
    manifest=dict(motion_source=spec,frames=4)
    valid=validate_body_attachment(tmp_path,manifest)
    assert valid['family']=='robot' and valid['sha256']
    with pytest.raises(ValueError,match='registry'):
        validate_body_attachment(tmp_path,dict(motion_source=dict(spec,model_path='/private/model.xml')))


def test_recorded_joint_order_is_names_not_array_position(tmp_path,monkeypatch):
    row,spec,raw=fixture(tmp_path,monkeypatch)
    roots=['root_x','root_y','root_z','root_qw','root_qx','root_qy','root_qz']
    q=np.concatenate([raw['root_position'],raw['root_quaternion_wxyz'],raw['joint_positions']],axis=1)
    names=roots+raw['joint_names'].tolist()
    a=robot.recorded_state_arrays(q,names,raw['timestamps_s'],row)
    permutation=np.random.default_rng(8).permutation(len(names))
    b=robot.recorded_state_arrays(q[:,permutation],[names[i] for i in permutation],raw['timestamps_s'],row)
    for k in a:np.testing.assert_array_equal(a[k],b[k])


def test_preflight_asks_for_source_robot_size_not_human_template(tmp_path,monkeypatch):
    row,spec,_=fixture(tmp_path,monkeypatch)
    from arenalab_workbench import source_requirements as req
    monkeypatch.setattr(req,'_robot_height',lambda r:1.0 if r['name']==row['name'] else 1.7)
    result=req.motion_preflight(dict(motion_source=spec,source_height_m=99),dict(row,name='large'))
    assert result['source_policy']=='supplied-robot'
    assert result['generated_human_model'] is None
    assert result['morphology']['source_height_m']==1.
    assert result['scale_choice_required']
    with pytest.raises(ValueError,match='SCALE_CHOICE_REQUIRED'):req.resolve_scale(None,result)
    assert req.resolve_scale('preserve',result)=='preserve'
    assert req.resolve_scale('match_robot',result)=='match_robot'
    comparable=req.motion_preflight(dict(motion_source=spec),row)
    assert not comparable['scale_choice_required']
    assert req.resolve_scale(None,comparable)=='preserve'
    assert comparable['morphology']['scale_factor']==1.
    from arenalab_workbench.motion_sources import describe_source
    description=describe_source(spec)
    assert description['route']=='supplied-robot' and description['frames']==4
    assert description['neutral_mesh_height_m']==1.






def test_attach_complete_source_keeps_registered_observations_and_clock(tmp_path,monkeypatch):
    _,spec,raw=fixture(tmp_path,monkeypatch)
    from arenalab_workbench.attach_robot_motion import attach
    from arenalab_workbench.dataset_io.ingest import prepare_body
    from arenalab_workbench.motion_sources import sha256
    obs=tmp_path/'observed.npz';p=np.ones((4,2,3))*.2
    np.savez_compressed(obs,wrist_position=p,palm_position=p,head_position=np.ones((5,3)),
        head_rotation=np.broadcast_to(np.eye(3),(5,3,3)),palm_forward=np.broadcast_to([1.,0,0],p.shape),
        palm_normal=np.broadcast_to([0.,0,1],p.shape),timestamps_s=raw['timestamps_s'],floor_z=np.array(0.))
    record=dict(body_adapter='head-wrists-v1',public=dict(id='fixture',dataset='Recorded',end_effector='none',video=dict(url='/assets/character.mp4',start_seconds=0)),
                source_row=dict(source=str(obs),frames=4,video=None))
    saved=json.dumps(record,sort_keys=True);digest=sha256(obs)
    upgraded=attach(record,spec)
    assert upgraded['public']['has_supplied_body']
    assert upgraded['public']['video'] is None
    assert upgraded['public']['auxiliary_video']==record['public']['video']
    assert json.dumps(record,sort_keys=True)==saved
    work=prepare_body(upgraded,tmp_path/'registered')
    manifest=json.loads((work/'input_manifest.json').read_text())
    assert source_route(manifest)['route']=='supplied-robot'
    normalized=prepare_body_source(work,tmp_path/'normalized-registered')
    with np.load(normalized/'native_source.npz') as z:np.testing.assert_array_equal(z['wrist_position_world_m'],p)
    assert sha256(obs)==digest
    changed=dict(spec,frames=3)
    with pytest.raises(ValueError,match='cover'):attach(record,changed)
    video=tmp_path/'robot.mp4';video.write_bytes(b'preview-integrity-fixture')
    preview=dict(kind='robot_motion',robot_id='fixture',source_sha256=sha256(spec['path']),frames=4,
                 retargeted=False,source_trajectory_modified=False,source_timestamps_s=raw['timestamps_s'].tolist(),
                 path=str(video),video_sha256=sha256(video))
    ready=attach(record,spec,preview=preview,preview_url='/assets/robot.mp4')
    assert ready['public']['video']['robot_id']=='fixture'
    assert ready['public']['video']['kind']=='robot_motion'
    with pytest.raises(ValueError,match='exact original'):
        attach(record,spec,preview=dict(preview,robot_id='different-target'),preview_url='/assets/robot.mp4')


def test_robot_scene_does_not_reuse_sparse_reconstruction_scene(monkeypatch):
    from arenalab_workbench import unified_scene
    from arenalab_workbench.dataset_io import storage
    monkeypatch.setattr(storage, 'entry', lambda _: {})
    old = unified_scene.scene_key('same-recording', 'preserve', 'target', ['trajectory'])
    state = dict(format='robot_joint_state', robot_id='source', sha256='original')
    monkeypatch.setattr(storage, 'entry', lambda _: dict(motion_source=state))
    new = unified_scene.scene_key('same-recording', 'preserve', 'target', ['trajectory'])
    assert new != old
    state['sha256'] = 'new-source'
    assert unified_scene.scene_key('same-recording', 'preserve', 'target', ['trajectory']) != new


