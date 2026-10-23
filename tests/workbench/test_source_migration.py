import copy
import json
import numpy as np
import pytest

from arenalab_workbench.source_migration import replace_supplied_source


@pytest.fixture
def replacement(tmp_path, monkeypatch):
    from arenalab_workbench import motion_sources
    path = tmp_path / 'robot.npz'
    np.savez(path, joint_positions=np.zeros((299, 29)), timestamps_s=np.arange(299) / 30.)
    spec = dict(format='robot_joint_state', family='robot', robot_id='g1',
                path=str(path), sha256='new-native-hash', evidence='simulation_fk')
    monkeypatch.setattr(motion_sources, 'validate_body_attachment', lambda package, manifest: spec.copy())
    record = dict(public=dict(id='clip', dataset='Example', title='An action',
        source_body_family='smplx', source_evidence='reconstructed', duration=301 / 30.,
        scene_url='/assets/old-human-result/scene.json',
        source_motion=dict(source_family='smplx'), video=dict(url='/assets/original-character.mp4')),
        source_row=dict(id='clip', source='/original/human.npz', frames=301, fps=30., video=None),
        motion_source=dict(format='smpl_parameters', family='smplx', path='/original/human.npz', sha256='old-hash'),
        manifest={}, body_adapter='supplied-body')
    manifest = dict(frames=299, source_stage='post_rl', end_effector='none', motion_source=spec)
    return record, tmp_path, manifest


def test_explicit_source_change_rebinds_clock_and_invalidates_old_preview(replacement):
    record, folder, manifest = replacement
    original = copy.deepcopy(record)
    changed = replace_supplied_source(record, folder, manifest)
    assert record == original
    assert changed['source_row']['frames'] == 299
    assert changed['source_row']['source'].endswith('/robot.npz')
    assert changed['public']['duration'] == pytest.approx(298 / 30.)
    assert changed['public']['source_body_family'] == 'robot'
    assert changed['public']['source_format'] == 'robot_joint_state'
    assert changed['public']['end_effector'] == 'none'
    assert changed['public']['source_stage'] == 'post_rl'
    assert changed['public']['source_evidence'] == 'simulation_fk'
    assert 'scene_url' not in changed['public']
    assert 'source_motion' not in changed['public']
    assert changed['public']['video'] is None
    assert changed['public']['auxiliary_video']['url'].endswith('original-character.mp4')
    assert changed['source_revision']['previous_source'] == original['motion_source']


def test_source_change_refuses_unsynchronized_independent_observations(replacement):
    record, folder, manifest = replacement
    record['source_row']['observations_source'] = '/original/observations.npz'
    with pytest.raises(ValueError, match='independent observations'):
        replace_supplied_source(record, folder, manifest)


def test_source_change_rejects_previous_or_target_robot_preview(replacement):
    record, folder, manifest = replacement
    preview = dict(source_sha256='old-hash', source_robot_id='g1', frames=299, fps=30., world_up='z',
                   retargeted=False, source_trajectory_modified=False)
    with pytest.raises(ValueError, match='exact native source'):
        replace_supplied_source(record, folder, manifest, preview=preview)


def test_exact_source_preview_is_exposed(replacement):
    record, folder, manifest = replacement
    preview = dict(source_sha256='new-native-hash', source_robot_id='g1', frames=299, fps=30., world_up='z',
                   retargeted=False, source_trajectory_modified=False)
    result = replace_supplied_source(record, folder, manifest, preview=preview)
    assert result['public']['source_motion'] == preview


@pytest.mark.parametrize('key,value', [('fps', 60.), ('world_up', 'y'), ('frames', 301)])
def test_preview_must_share_native_clock_and_basis(replacement, key, value):
    record, folder, manifest = replacement
    preview = dict(source_sha256='new-native-hash', source_robot_id='g1', frames=299, fps=30., world_up='z',
                   retargeted=False, source_trajectory_modified=False)
    preview[key] = value
    with pytest.raises(ValueError, match='exact native source'):
        replace_supplied_source(record, folder, manifest, preview=preview)


def test_stage_precedence_and_no_hand_mode_are_taken_from_new_source(replacement):
    record, folder, manifest = replacement
    record['public']['end_effector'] = 'wuji'
    manifest['motion_source']['source_stage'] = 'native-stage'
    result = replace_supplied_source(record, folder, manifest)
    assert result['public']['source_stage'] == result['source_revision']['source_stage'] == 'native-stage'
    assert result['public']['end_effector'] == 'none'
    result['source_revision']['previous_source']['family'] = 'changed-copy'
    assert record['motion_source']['family'] == 'smplx'


@pytest.mark.parametrize('object_frames', [299, 301])
def test_migration_uses_real_attachment_pair_validation(tmp_path, monkeypatch, object_frames):
    from tests.workbench.test_source_attachment_contract import source, objects, attach
    manifest, times = source(tmp_path, monkeypatch, 'robot', 299)
    attach(tmp_path, manifest, objects(np.arange(object_frames)/30.))
    record = dict(public=dict(id='clip', owner='private-workspace', uploaded=True),
                  source_row=dict(id='clip', source='/old/human.npz', frames=301, fps=30.),
                  motion_source=dict(path='/old/human.npz', family='smplx', sha256='old'))
    raw = (tmp_path/'motion.npz').read_bytes(); object_raw = (tmp_path/'objects.npz').read_bytes()
    if object_frames != 299:
        with pytest.raises(ValueError, match='exact source motion clock'):
            replace_supplied_source(record, tmp_path, manifest)
    else:
        result = replace_supplied_source(record, tmp_path, manifest)
        assert result['source_row']['frames'] == 299
        assert result['motion_source']['objects_path'] == str(tmp_path/'objects.npz')
        assert result['public']['owner'] == 'private-workspace' and result['public']['uploaded']
    assert (tmp_path/'motion.npz').read_bytes() == raw
    assert (tmp_path/'objects.npz').read_bytes() == object_raw


@pytest.mark.parametrize('expected', ['old-hash', 'stale-hash'])
def test_cli_preserves_private_records_original_registry_and_files(replacement, monkeypatch, expected):
    from arenalab_workbench.source_migration import main
    record, folder, manifest = replacement
    record['public'].update(owner=None, uploaded=False)
    private = dict(public=dict(id='private-record', owner='not-public', uploaded=True),
                   private_annotation='keep verbatim', source_row=dict(source='/private/original.npz'))
    registry = folder/'registry.json'; registry.write_text(json.dumps({'clip': record, 'private-record': private}))
    original = registry.read_bytes(); motion = (folder/'robot.npz').read_bytes()
    (folder/'recording.json').write_text(json.dumps(manifest))
    plan = folder/'plan.json'; plan.write_text(json.dumps(dict(replacements=[dict(
        recording_id='clip', expected_source_sha256=expected, package=str(folder))])))
    output = folder/'snapshot.json'
    monkeypatch.setattr('sys.argv', ['source_migration', '--registry', str(registry), '--plan', str(plan), '--output', str(output)])
    if expected != 'old-hash':
        with pytest.raises(ValueError, match='Registered source changed'):
            main()
        assert not output.exists()
    else:
        main()
        result = json.loads(output.read_text())
        assert set(result) == {'clip', 'private-record'}
        assert result['private-record'] == private
        assert result['clip']['public']['owner'] is None and not result['clip']['public']['uploaded']
        snapshot = output.read_bytes()
        with pytest.raises(FileExistsError): main()
        assert output.read_bytes() == snapshot
    assert registry.read_bytes() == original
    assert (folder/'robot.npz').read_bytes() == motion
