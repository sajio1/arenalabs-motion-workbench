import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arenalab_workbench import unified_conversion as conversion
from greenwich_motion_sdk.motion import MotionClip


def clip_fixture(frames=9, fps=90):
    names=['Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot']
    rest=np.array([[0.,0,0],[0,160,0],[30,100,0],[-30,100,0],[10,0,0],[-10,0,0]])
    positions=np.broadcast_to(rest,(frames,6,3)).copy()
    positions[:,:,0]+=np.arange(frames)[:,None]
    return MotionClip(names,np.array([-1,0,0,0,0,0]),rest,
                      np.broadcast_to(np.eye(3),(frames,6,3,3)).copy(),positions,
                      np.arange(frames)/fps,dict(format='test-capture',ground_y_cm=0.))


def test_explicit_smpl_bypasses_reconstruction(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(conversion,'_provided',lambda spec,folder,base=None:calls.append((spec,folder)) or folder)
    monkeypatch.setattr(conversion,'three_point_work',lambda *a:pytest.fail('Supplied SMPL must not reconstruct'))
    spec=dict(family='smplh',format='smpl_decoded',path='/private/body.npz')
    assert conversion.prepare_sources(dict(motion_source=spec),tmp_path)==[tmp_path/'source']
    assert calls[0][0]==spec


def test_smpl_manifest_uses_model_adapter_and_profile_clock_without_legacy_fk(monkeypatch,tmp_path):
    np.savez(tmp_path/'body.npz',poses=np.zeros((12,156)),trans=np.zeros((12,3)),mocap_frame_rate=60.)
    (tmp_path/'profile.json').write_text(json.dumps(dict(frame=dict(basis_to_yup=conversion.C.tolist(),cm_per_unit=1.,ground_y_cm=3.))))
    (tmp_path/'sources.json').write_text(json.dumps(dict(schema='alphamotion.sources.v1',sources=[dict(id='sample',format='smpl',path='body.npz',options_file='profile.json')])))
    calls=[]
    monkeypatch.setattr(conversion,'_provided',lambda spec,folder,base=None:calls.append(spec) or folder)
    import greenwich_motion_sdk.adapters as adapters
    monkeypatch.setattr(adapters,'load_motion',lambda *a,**k:pytest.fail('Do not invoke old SMPL FK adapter'))
    conversion.prepare_sources(dict(source=str(tmp_path/'sources.json')),tmp_path/'out')
    assert len(calls)==1
    assert calls[0]['family']=='smplh'
    assert calls[0]['stride']==2 and calls[0]['frames']==6
    assert calls[0]['units']=='cm' and calls[0]['world_up']=='z'
    assert calls[0]['floor_z_m']==.03


def test_non_smpl_canonical_routes_three_points_at_original_clock(monkeypatch,tmp_path):
    original=clip_fixture();original.save(tmp_path/'canonical.npz')
    (tmp_path/'sources.json').write_text(json.dumps(dict(schema='alphamotion.sources.v1',sources=[dict(id='sample',format='canonical',path='canonical.npz',roles=original.names)])))
    calls=[]
    monkeypatch.setattr(conversion,'three_point_work',lambda clip,roles,folder:calls.append((clip,roles)) or folder)
    conversion.prepare_sources(dict(source=str(tmp_path/'sources.json')),tmp_path/'out')
    assert len(calls)==1 and calls[0][1]==list(range(6))
    np.testing.assert_array_equal(calls[0][0].timestamps_s,original.timestamps_s[::3])
    np.testing.assert_array_equal(calls[0][0].world_position_cm,original.world_position_cm[::3])


def test_traceable_smpl_canonical_metadata_bypasses_three_points(monkeypatch,tmp_path):
    original=clip_fixture()
    original.metadata['motion_source']=dict(family='smplh',format='smpl_decoded',path='native_h.npz')
    original.save(tmp_path/'canonical.npz')
    (tmp_path/'sources.json').write_text(json.dumps(dict(schema='alphamotion.sources.v1',sources=[dict(id='sample',format='canonical',path='canonical.npz')])))
    calls=[]
    monkeypatch.setattr(conversion,'_provided',lambda spec,folder,base=None:calls.append((spec,base)) or folder)
    monkeypatch.setattr(conversion,'three_point_work',lambda *a:pytest.fail('Traceable SMPL must not reconstruct'))
    conversion.prepare_sources(dict(source=str(tmp_path/'sources.json')),tmp_path/'out')
    assert calls[0][0]==original.metadata['motion_source'] and calls[0][1]==tmp_path


def test_three_point_stage_preserves_samples_and_invokes_native_h_once(monkeypatch,tmp_path):
    assets=tmp_path/'assets';assets.mkdir();(assets/'model.npz').write_bytes(b'private model placeholder')
    (assets/'egoallo_checkpoint_april13').mkdir()
    monkeypatch.setattr(conversion,'paths',SimpleNamespace(egoallo=assets))
    import arenalab_workbench.body_inference as inference
    calls=[]
    monkeypatch.setattr(inference,'run',lambda path:calls.append(path) or path)
    monkeypatch.setattr(conversion,'normalized_source',lambda path:path/'normalized')
    original=clip_fixture(frames=4,fps=30);output=tmp_path/'work'
    assert conversion.three_point_work(original,list(range(6)),output)==output/'normalized'
    assert calls==[output]
    with np.load(output/'observations.npz') as obs:
        np.testing.assert_array_equal(obs['wrist_position'],original.world_position_cm[:,2:4]/100@conversion.C)
        np.testing.assert_array_equal(obs['head_position'][1:],original.world_position_cm[:,1]/100@conversion.C)
        np.testing.assert_array_equal(obs['timestamps_s'],original.timestamps_s)
    assert not (output/'canonical.npz').exists()  # Body is not fabricated by this stage.
    original_bytes=(output/'observations.npz').read_bytes()
    with pytest.raises(FileExistsError):conversion.three_point_work(original,list(range(6)),output)
    assert (output/'observations.npz').read_bytes()==original_bytes


def test_smpl_missing_clock_fails_before_generation(tmp_path):
    path=tmp_path/'body.npz';np.savez(path,poses=np.zeros((3,72)),trans=np.zeros((3,3)))
    with pytest.raises(ValueError,match='clock'):
        conversion._smpl_manifest_spec(path,{}, {})
