"""Native customer adapter contracts; no neural weights, cloud or GPU required."""
from pathlib import Path
import json
import shutil
import numpy as np
import pytest

mj=pytest.importorskip('mujoco')
from arenalab_workbench.dataset_io.native_reference import build_native_reference
from arenalab_workbench.dataset_io.native_reference_mujoco import load, load_model, apply_frame, validate, model_signature

XML='''<mujoco><compiler angle="radian"/><worldbody><body name="base" pos="0 0 1"><freejoint name="root"/><geom type="box" size=".1 .1 .1"/><body name="arm" pos=".2 0 .1"><joint name="hinge" type="hinge" axis="0 1 0" range="-1 1"/><geom type="capsule" size=".04 .15"/><body name="tool" pos=".1 0 .1"><joint name="slide" type="slide" axis="1 0 0" range="-.1 .1"/><geom type="sphere" size=".05"/></body></body></body></worldbody></mujoco>'''


@pytest.fixture
def native(tmp_path):
    model_path=tmp_path/'robot.xml';model_path.write_text(XML)
    model=mj.MjModel.from_xml_path(str(model_path));q=np.tile(model.qpos0,(4,1));q[:,0]=[0,.02,.04,.07]
    q[:,3:7]=np.array([[1,0,0,0],[-1,0,0,0],[np.cos(.05),0,0,np.sin(.05)],[-np.cos(.06),0,0,-np.sin(.06)]])
    q[:,7]=[0,.1,.12,.15];q[:,8]=[0,.01,.02,.03]
    motion=tmp_path/'motion.npz';np.savez_compressed(motion,qpos=q,qpos_world=q,fps=30,timestamps_s=123+np.arange(4)/30)
    playback=dict(frames=4,fps=30,joints=dict(names=['slide','hinge','external_finger'],values=np.c_[q[:,8],q[:,7],np.linspace(0,.2,4)].tolist(),unit='rad',units=['m','rad','rad']))
    return model_path,motion,playback,q


def test_native_export_preserves_pose_and_supplied_hands(native,tmp_path):
    model,motion,playback,raw=native;original=motion.read_bytes();out=tmp_path/'export'
    result=build_native_reference(motion,model,out,playback=playback,robot_id='test')
    meta,frames=load(out);context=load_model(out,model,body_only=True)
    assert result['frames']==4 and meta['model']['included'] is False
    assert frames['joint_names'].tolist()==['hinge','slide']
    assert frames['additional_joint_names'].tolist()==['external_finger']
    assert meta['additional_joints']['units']==['rad'] and not meta['additional_joints']['model_validated_at_export']
    assert np.allclose(frames['source_timestamp_s'],123+frames['timestamp_s'])
    assert np.allclose(frames['native_qpos'][:,:3],raw[:,:3])
    assert np.all(np.sum(frames['root_quaternion_wxyz'][:-1]*frames['root_quaternion_wxyz'][1:],axis=-1)>=0)
    assert motion.read_bytes()==original
    for i in range(3):
        state=apply_frame(context,i);nextq=state.qpos.copy();mj.mj_integratePos(context['model'],nextq,state.qvel,1/30)
        assert np.allclose(nextq,frames['native_qpos'][i+1],atol=1e-12)
    assert validate(out,model,body_only=True)['kinematic_replay_passed']
    assert not meta['physics']['dynamic_tracking_validated']
    assert (out/'native_reference_mujoco.py').is_file()
    assert not any(out.rglob('*.xml')) and not any(out.rglob('*.urdf'))


def test_relocated_model_works_but_changed_kinematics_rejected(native,tmp_path):
    model,motion,playback,_=native;out=tmp_path/'export';build_native_reference(motion,model,out,playback=playback)
    local=tmp_path/'customer';local.mkdir();relocated=local/'different-name.xml';relocated.write_text('\n'+model.read_text())
    assert validate(out,relocated,body_only=True)['position_max_m']<1e-12
    relocated.write_text(XML.replace('pos=".2 0 .1"','pos=".3 0 .1"'))
    with pytest.raises(ValueError,match='kinematics do not match'):load_model(out,relocated,body_only=True)


def test_playback_mismatch_and_wrong_clock_rejected(native,tmp_path):
    model,motion,playback,raw=native;playback['joints']['values'][1][1]+=.5
    with pytest.raises(ValueError,match='scalar joint differs'):build_native_reference(motion,model,tmp_path/'bad-joints',playback=playback)
    np.savez(motion,qpos=raw,fps=60)
    with pytest.raises(ValueError,match='30 Hz'):build_native_reference(motion,model,tmp_path/'bad-clock')


def test_package_tamper_and_escape_rejected(native,tmp_path):
    model,motion,playback,_=native;out=tmp_path/'export';build_native_reference(motion,model,out,playback=playback)
    manifest=out/'manifest.json';meta=json.loads(manifest.read_text());meta['data']='../motion.npz';manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='escapes'):load(out)
    meta['data']='reference.npz';manifest.write_text(json.dumps(meta));(out/'reference.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):load(out)


def test_ambiguous_native_arrays_and_nonunit_quaternion_rejected(native,tmp_path):
    model,motion,_,raw=native;other=raw.copy();other[0,0]+=.1
    np.savez(motion,qpos=raw,qpos_world=other,fps=30)
    with pytest.raises(ValueError,match='Ambiguous'):build_native_reference(motion,model,tmp_path/'ambiguous')
    raw[0,3:7]=0;np.savez(motion,qpos=raw,fps=30)
    with pytest.raises(ValueError,match='non-unit'):build_native_reference(motion,model,tmp_path/'zero-quaternion')


def test_source_fk_must_match_selected_model(native,tmp_path):
    model,motion,_,raw=native
    np.savez(motion,qpos=raw,fps=30,native_body_position_m=np.zeros((4,1,3)),descriptor_body_ids=np.array([1]))
    with pytest.raises(ValueError,match='saved position FK differs'):build_native_reference(motion,model,tmp_path/'wrong-source')


def test_full_native_mapping_is_by_names_not_array_order(tmp_path):
    # Two sibling hinges can be reordered by a local loader without changing FK.
    prefix='<mujoco><worldbody><body name="base"><freejoint name="root"/><geom type="sphere" size=".1"/>'
    first='<body name="first" pos=".2 0 0"><joint name="a" axis="1 0 0"/><geom type="sphere" size=".1"/></body>'
    second='<body name="second" pos="0 .2 0"><joint name="b" axis="0 1 0"/><geom type="sphere" size=".1"/></body>'
    suffix='</body></worldbody></mujoco>'
    a=tmp_path/'a.xml';b=tmp_path/'b.xml';a.write_text(prefix+first+second+suffix);b.write_text(prefix+second+first+suffix)
    m=mj.MjModel.from_xml_path(str(a));q=np.tile(m.qpos0,(2,1));q[:,7:]=[[.1,.2],[.2,.4]]
    motion=tmp_path/'motion.npz';np.savez(motion,qpos=q,fps=30)
    build_native_reference(motion,a,tmp_path/'export')
    assert model_signature(m)==model_signature(mj.MjModel.from_xml_path(str(b)))
    assert validate(tmp_path/'export',b)['position_max_m']<1e-12


def test_unknown_world_convention_rejected(native,tmp_path):
    model,motion,_,_=native;out=tmp_path/'export';build_native_reference(motion,model,out)
    path=out/'manifest.json';meta=json.loads(path.read_text());meta['world']['quaternion_order']='xyzw';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='world conventions'):load(out)


def test_empty_external_channels_are_explicit_and_same_schema(native,tmp_path):
    model,motion,_,_=native;out=tmp_path/'export';build_native_reference(motion,model,out)
    meta,frames=load(out)
    assert frames['additional_joint_position'].shape==(4,0)
    assert frames['additional_joint_names'].tolist()==frames['additional_joint_units'].tolist()==[]
    assert meta['additional_joints']['names']==[]


def test_native_layout_overlap_and_redundant_state_mismatch_rejected(native,tmp_path):
    from arenalab_workbench.dataset_io.native_reference_mujoco import sha
    model,motion,_,_=native;out=tmp_path/'export';build_native_reference(motion,model,out)
    manifest=out/'manifest.json';meta=json.loads(manifest.read_text())
    meta['native_joints'][2]['native_qpos_address']=7;manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='partition'):load(out)
    meta['native_joints'][2]['native_qpos_address']=8
    with np.load(out/'reference.npz') as z:arrays=dict(z)
    arrays['joint_position'][0,0]+=.2;np.savez_compressed(out/'reference.npz',**arrays);meta['data_sha256']=sha(out/'reference.npz');manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='arrays disagree'):load(out)


def test_source_clock_must_be_30_hz(native,tmp_path):
    model,motion,_,raw=native
    np.savez(motion,qpos=raw,fps=30,timestamps_s=100+np.arange(4)/60)
    with pytest.raises(ValueError,match='Source timestamps'):build_native_reference(motion,model,tmp_path/'bad-source-clock')


def test_additional_components_required_and_applied_by_name(native,tmp_path):
    model,motion,playback,_=native;out=tmp_path/'export';build_native_reference(motion,model,out,playback=playback)
    assert load_model(out,model)['body_only']
    with pytest.raises(ValueError,match='missing=.*external_finger'):load_model(out,model,body_only=False)
    assembly=tmp_path/'assembly.xml'
    attachment='<body name="finger" pos="0 0 .1"><joint name="external_finger" type="hinge" axis="1 0 0"/><geom type="sphere" size=".02"/></body>'
    assembly.write_text(XML.replace('</body></body></body></worldbody>',attachment+'</body></body></body></worldbody>'))
    context=load_model(out,assembly,body_only=False);state=apply_frame(context,2);qa=context['model'].jnt_qposadr[context['model'].joint('external_finger').id]
    assert state.qpos[qa]==playback['joints']['values'][2][2]
    report=validate(out,assembly,body_only=False)
    assert report['all_reference_channels_applied'] and report['applied_additional_joint_names']==['external_finger']
    assert report['additional_joint_position_max_abs']==0 and not report['additional_link_fk_reference_available']
    partial=validate(out,model,body_only=True)
    assert not partial['all_reference_channels_applied'] and partial['unapplied_additional_joint_names']==['external_finger']
    assembly.write_text(assembly.read_text().replace('name="external_finger"','name="wrong_finger"'))
    with pytest.raises(ValueError,match='joint channels differ'):load_model(out,assembly,body_only=False)


def test_explicit_local_mimic_uses_reference_offsets_and_velocity(native,tmp_path):
    model,motion,playback,_=native;out=tmp_path/'export';build_native_reference(motion,model,out,playback=playback)
    attachment='<body name="finger" pos="0 0 .1"><joint name="external_finger" type="hinge" axis="1 0 0" ref=".2"/><geom type="sphere" size=".02"/><body name="coupled"><joint name="coupled_joint" axis="1 0 0" ref=".3"/><geom type="sphere" size=".02"/></body></body>'
    assembly=tmp_path/'assembly.xml';text=XML.replace('</body></body></body></worldbody>',attachment+'</body></body></body></worldbody>')
    text=text.replace('</mujoco>','<equality><joint name="explicit_mimic" joint1="coupled_joint" joint2="external_finger" polycoef=".1 -2 .5 0 0"/></equality></mujoco>');assembly.write_text(text)
    context=load_model(out,assembly,body_only=False);state=apply_frame(context,2);m=context['model'];slave=m.joint('coupled_joint').id;driver=m.joint('external_finger').id
    x=playback['joints']['values'][2][2]-.2
    assert state.qpos[m.jnt_qposadr[slave]]==pytest.approx(.3+.1-2*x+.5*x*x)
    assert state.qvel[m.jnt_dofadr[slave]]==pytest.approx((-2+x)*state.qvel[m.jnt_dofadr[driver]])
    assert validate(out,assembly,body_only=False)['derived_mimic_joint_names']==['coupled_joint']
    assembly.write_text(text.replace('name="explicit_mimic"','name="explicit_mimic" active="false"'))
    with pytest.raises(ValueError,match='without a resolvable explicit'):load_model(out,assembly,body_only=False)


def test_default_body_replay_does_not_require_optional_hand_model(native,tmp_path):
    model,motion,playback,_=native;out=tmp_path/'export';build_native_reference(motion,model,out,playback=playback)
    context=load_model(out,model);apply_frame(context,2)
    report=validate(out,model)
    assert report['body_only'] and report['kinematic_replay_passed']
    assert report['unapplied_additional_joint_names']==['external_finger']
    assert not report['all_reference_channels_applied']
    with pytest.raises(ValueError,match='missing='):load_model(out,model,body_only=False)
