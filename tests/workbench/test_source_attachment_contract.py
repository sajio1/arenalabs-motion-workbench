"""Reject mispaired source objects before registration, without decoding or GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from arenalab_workbench.motion_sources import validate_body_attachment


def source(tmp_path, monkeypatch, family, frames, *, start=0.):
    times = start + np.arange(frames, dtype=np.float64) / 30
    spec = dict(path='motion.npz', world_up='z', units='m', floor_z_m=0.,
                evidence='synthetic', source_stage='source-fixture')
    if family == 'robot':
        mj = pytest.importorskip('mujoco')
        from arenalab_workbench import robot_motion_source
        from arenalab_motion import prior
        model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><freejoint/><geom type="sphere" size=".1"/>
            <body pos="0 0 .2"><joint name="arm" type="hinge"/>
              <geom type="capsule" size=".03 .1"/></body>
          </body></worldbody></mujoco>''')
        monkeypatch.setattr(robot_motion_source, 'source_robot', lambda spec: {'name': 'fixture'})
        monkeypatch.setattr(prior, 'load_target', lambda row: {'model': model})
        raw = dict(joint_names=np.array(['arm']), joint_positions=np.zeros((frames, 1)),
                   root_position=np.zeros((frames, 3)),
                   root_quaternion_wxyz=np.tile([1., 0., 0., 0.], (frames, 1)))
        spec.update(format='robot_joint_state', family='robot', robot_id='fixture')
    else:
        raw = dict(poses=np.zeros((frames, 165)), trans=np.zeros((frames, 3)),
                   betas=np.zeros(10), model_type=np.array('smplx'))
        spec.update(format='smpl_parameters', family='smplx')
    np.savez(tmp_path / 'motion.npz', timestamps_s=times, **raw)
    return dict(frames=frames, end_effector='none', motion_source=spec), times


def objects(times):
    return dict(object_ids=np.array(['object']), timestamps_s=times.copy(),
                positions_world_m=np.zeros((len(times), 1, 3)),
                rotations_world=np.tile(np.eye(3), (len(times), 1, 1, 1)),
                object_0_vertices_local_m=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                object_0_faces=np.array([[0, 1, 2]], dtype=np.int32))


def attach(tmp_path, manifest, data):
    path = tmp_path / 'objects.npz'
    np.savez(path, **data)
    manifest['motion_source']['objects_path'] = path.name
    return path


def test_human_301_does_not_accept_299_frame_paired_robot_objects(tmp_path, monkeypatch):
    manifest, _ = source(tmp_path, monkeypatch, 'smplx', 301)
    path = attach(tmp_path, manifest, objects(np.arange(299) / 30))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='exact source motion clock'):
        validate_body_attachment(tmp_path, manifest)
    assert path.read_bytes() == original


@pytest.mark.parametrize('family', ['smplx', 'robot'])
def test_matched_299_frame_source_and_objects_validate_without_repair(tmp_path, monkeypatch, family):
    manifest, times = source(tmp_path, monkeypatch, family, 299)
    path = attach(tmp_path, manifest, objects(times))
    original = path.read_bytes()
    result = validate_body_attachment(tmp_path, manifest)
    assert result['objects_path'] == str(path)
    assert result['family'] == family
    assert path.read_bytes() == original


@pytest.mark.parametrize('family', ['smplx', 'robot'])
def test_no_hand_source_without_objects_still_validates(tmp_path, monkeypatch, family):
    manifest, _ = source(tmp_path, monkeypatch, family, 4)
    result = validate_body_attachment(tmp_path, manifest)
    assert 'objects_path' not in result
    assert manifest['end_effector'] == 'none'


def test_recorded_skeleton_keeps_legitimate_masked_extra_joint_nans(tmp_path):
    from tests.workbench.test_recorded_surface import recorded
    spec, raw, _, _, _ = recorded(tmp_path)
    raw['source_joints'][:, 22:25] = np.nan
    raw['source_joints_mask'][:, 22:25] = False
    times = np.arange(len(raw['poses'])) / 30
    raw['timestamps_s'] = times
    spec.pop('stride')
    spec['path'] = 'motion.npz'
    np.savez(tmp_path / 'motion.npz', **raw)
    manifest = dict(frames=len(times), end_effector='none', motion_source=spec)
    attach(tmp_path, manifest, objects(times))
    original = (tmp_path / 'motion.npz').read_bytes()
    validated = validate_body_attachment(tmp_path, manifest)
    assert validated['format'] == 'recorded_skeleton_surface'
    assert (tmp_path / 'motion.npz').read_bytes() == original


@pytest.mark.parametrize('family', ['smplx', 'robot'])
@pytest.mark.parametrize('broken', ['offset_clock', 'nan_clock', 'nan_position', 'non_rotation',
                                  'position_shape', 'nan_mesh', 'fractional_faces', 'outside_face',
                                  'missing_mesh', 'duplicate_ids'])
def test_both_attachment_routes_reject_invalid_object_containers(tmp_path, monkeypatch, family, broken):
    manifest, times = source(tmp_path, monkeypatch, family, 4)
    data = objects(times)
    if broken == 'offset_clock':data['timestamps_s'] += 1 / 30
    elif broken == 'nan_clock':data['timestamps_s'][1] = np.nan
    elif broken == 'nan_position':data['positions_world_m'][1, 0, 0] = np.nan
    elif broken == 'non_rotation':data['rotations_world'][1, 0, 0, 0] = -1.
    elif broken == 'position_shape':data['positions_world_m'] = data['positions_world_m'][:, 0]
    elif broken == 'nan_mesh':data['object_0_vertices_local_m'][0, 0] = np.nan
    elif broken == 'fractional_faces':data['object_0_faces'] = np.array([[0., 1., 1.5]])
    elif broken == 'outside_face':data['object_0_faces'][0, 2] = 3
    elif broken == 'missing_mesh':data.pop('object_0_faces')
    elif broken == 'duplicate_ids':data['object_ids'] = np.array(['object', 'object'])
    path = attach(tmp_path, manifest, data)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        validate_body_attachment(tmp_path, manifest)
    assert path.read_bytes() == original


def registration():
    path = Path(__file__).resolve().parents[2] / 'tools/register_motion_sources.py'
    spec = importlib.util.spec_from_file_location('source_registration_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('family', ['smplx', 'robot'])
def test_registration_uses_timestamp_span_and_actual_source_metadata(tmp_path, monkeypatch, family):
    folder = tmp_path / 'recording'; folder.mkdir()
    manifest, times = source(folder, monkeypatch, family, 299, start=7.)
    attach(folder, manifest, objects(times))
    manifest['motion_source'].update(source_stage='post_rl' if family == 'robot' else 'reconstruction')
    (folder / 'recording.json').write_text(json.dumps(manifest))
    sample = dict(id='example', title='Example', dataset='Fixture', family='A real task group',
                  directory='recording', source_stage='must-not-override-source',
                  motion_sha256=hashlib.sha256((folder / 'motion.npz').read_bytes()).hexdigest())
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps({'samples': [sample]}))
    row = registration().records(path, tmp_path)['example']
    assert row['public']['duration'] == pytest.approx(298 / 30)
    assert row['public']['duration'] != pytest.approx(299 / 30)
    assert row['public']['source_family'] == row['public']['source_body_family'] == family
    assert row['public']['source_format'] == manifest['motion_source']['format']
    assert row['public']['source_stage'] == manifest['motion_source']['source_stage']
    assert row['public']['source_evidence'] == 'synthetic'
    assert row['public']['family'] == 'A real task group'
    assert row['public']['end_effector'] == 'none'
    if family == 'robot':
        assert row['public']['source_robot_id'] == 'fixture'
        assert row['public']['motion_source_policy'] == 'supplied-robot-fullbody-v1'
        assert row['body_adapter'] == 'supplied-robot'
    else:
        assert 'source_robot_id' not in row['public']
        assert row['body_adapter'] == 'supplied-body'


def test_registration_rejects_bad_objects_before_returning_any_rows(tmp_path, monkeypatch):
    folder = tmp_path / 'recording'; folder.mkdir()
    manifest, _ = source(folder, monkeypatch, 'smplx', 301)
    attach(folder, manifest, objects(np.arange(299) / 30))
    (folder / 'recording.json').write_text(json.dumps(manifest))
    sample = dict(id='bad', title='Bad', dataset='Fixture', directory='recording',
                  motion_sha256=hashlib.sha256((folder / 'motion.npz').read_bytes()).hexdigest())
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps({'samples': [sample]}))
    with pytest.raises(ValueError, match='exact source motion clock'):
        registration().records(path, tmp_path)
