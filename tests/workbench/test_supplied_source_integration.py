"""Source revisions and morphology remain consistent through shared Grid entrypoints."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arenalab_workbench import motion_sources, source_identity, source_preview, source_requirements, unified_scene
from tests.workbench.test_motion_sources import fixture
from tests.workbench.test_recorded_surface import recorded


def scene_fixture(tmp_path, monkeypatch):
    spec, _ = fixture(tmp_path)
    row = dict(source=spec['path'], frames=3, fps=30)
    monkeypatch.setenv('ARENA_SMPLX_MODEL', str(tmp_path / 'model.npz'))
    monkeypatch.setattr(source_identity, 'registered_source', lambda _: (row, spec))
    return spec, row


@pytest.mark.parametrize('changed', ['state', 'objects', 'spec', 'clock', 'observations', 'model', 'decoder'])
def test_every_geometry_dependency_versions_shared_human_scene(tmp_path, monkeypatch, changed):
    spec, row = scene_fixture(tmp_path, monkeypatch)
    objects = tmp_path / 'objects.npz'; objects.write_bytes(b'original-objects')
    observations = tmp_path / 'observed.npz'; observations.write_bytes(b'original-observations')
    spec['objects_path'] = str(objects)
    row['observations_source'] = str(observations)
    before = unified_scene.scene_key('recording', 'preserve', 'a3', ['trajectory'])
    if changed == 'state':
        with np.load(spec['path']) as data: raw = {key:data[key] for key in data.files}
        raw['trans'][0, 0] += .01
        np.savez(spec['path'], **raw)
    elif changed == 'objects': objects.write_bytes(b'new-objects')
    elif changed == 'spec': spec.update(format='recorded_skeleton_surface', skeleton_alignment='model_pelvis', surface_method='skinning_joint_offsets_v1')
    elif changed == 'clock': row['source_window'] = dict(start_frame=3)
    elif changed == 'observations': observations.write_bytes(b'new-observations')
    elif changed == 'model':
        with (tmp_path / 'model.npz').open('ab') as file: file.write(b'new-model-version')
    else:
        sha = motion_sources.sha256
        monkeypatch.setattr(motion_sources, 'sha256', lambda path: 'new-decoder' if Path(path).name == 'motion_sources.py' else sha(path))
    after = unified_scene.scene_key('recording', 'preserve', 'a3', ['trajectory'])
    assert after != before
    assert after == unified_scene.scene_key('recording', 'preserve', 'x2', ['trajectory'])


def test_recorded_proxy_decoder_is_bound_and_sparse_reconstruction_stays_target_independent(tmp_path, monkeypatch):
    spec, row = scene_fixture(tmp_path, monkeypatch)
    spec['format'] = 'recorded_skeleton_surface'
    identity = source_identity.supplied_body_identity(spec, row)
    assert set(identity['decoder_sha256']) == {'motion_sources.py', 'recorded_surface.py'}
    assert identity['model_sha256'] == motion_sources.sha256(tmp_path / 'model.npz')
    spec['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='body hash changed'):
        source_identity.supplied_body_identity(spec, row)
    monkeypatch.setattr(source_identity, 'registered_source', lambda _: (row, None))
    expected = hashlib.sha256(b'recording:preserve:resident-v3').hexdigest()
    assert unified_scene.scene_key('recording', 'preserve', 'a3') == expected
    assert unified_scene.scene_key('recording', 'preserve', 'x2') == expected
    assert source_identity.supplied_body_identity(None, row) is None


@pytest.mark.parametrize('kind', ['scaled_smpl', 'recorded'])
def test_preflight_uses_actual_source_anatomy_without_inference(tmp_path, monkeypatch, kind):
    if kind == 'recorded':
        spec, raw, *_ = recorded(tmp_path)
        np.savez(tmp_path / 'motion.npz', **raw)
    else:
        spec, _ = fixture(tmp_path)
        spec.update(body_scale=.5, scale_origin='smpl_translation')
    monkeypatch.setenv('ARENA_SMPLX_MODEL', str(tmp_path / 'model.npz'))
    expected = motion_sources.describe_source(spec)['neutral_mesh_height_m']
    monkeypatch.setattr(source_requirements, '_robot_height', lambda _: expected * 2)
    plan = source_requirements.motion_preflight(dict(motion_source=spec, source_height_m=expected * 2), {})
    assert plan['morphology']['source_height_m'] == expected
    assert plan['scale_choice_required']
    with pytest.raises(ValueError, match='SCALE_CHOICE_REQUIRED'):
        source_requirements.resolve_scale(None, plan)
    assert source_requirements.resolve_scale('preserve', plan) == 'preserve'
    assert source_requirements.resolve_scale('match_robot', plan) == 'match_robot'
    # No inference or animation artifacts are produced by the morphology query.
    assert not (tmp_path / 'native_source.npz').exists()


def test_missing_supplied_model_does_not_substitute_unscaled_template(tmp_path, monkeypatch):
    spec, _ = fixture(tmp_path)
    monkeypatch.delenv('ARENA_SMPLX_MODEL', raising=False)
    monkeypatch.setattr(source_requirements, '_robot_height', lambda _: 1.7)
    plan = source_requirements.motion_preflight(dict(motion_source=spec), {})
    assert not plan['morphology']['available']
    with pytest.raises(ValueError, match='SCALE_ESTIMATE_UNAVAILABLE'):
        source_requirements.resolve_scale('match_robot', plan)


def test_recorded_surface_preview_has_explicit_proxy_provenance_and_original_vertices(tmp_path, monkeypatch):
    spec, raw, *_ = recorded(tmp_path)
    np.savez(tmp_path / 'motion.npz', **raw)
    output = motion_sources.prepare_supplied_source(spec, tmp_path / 'source', model_path=tmp_path / 'model.npz')
    monkeypatch.setattr(source_preview, 'paths', SimpleNamespace(store=tmp_path))
    preview = source_preview.export_source_preview(output, tmp_path / 'public')
    assert preview['source_format'] == 'recorded_skeleton_surface'
    assert 'approximate' in preview['source_surface']
    assert 'not original subject SMPL' in preview['scope']
    assert preview == source_preview.export_source_preview(output, tmp_path / 'public')
    with np.load(output / 'source_mesh.npz') as mesh:
        binary = tmp_path / preview['vertices_url'].removeprefix('/assets/')
        np.testing.assert_array_equal(np.fromfile(binary, dtype='<f4').reshape(mesh['vertices_world_m'].shape), mesh['vertices_world_m'])
    # Old immutable SMPL previews remain distinct even if geometry happens to match.
    provenance = json.loads((output / 'source_provenance.json').read_text())
    provenance['source_format'] = 'smpl_parameters'
    (output / 'source_provenance.json').write_text(json.dumps(provenance))
    original = source_preview.export_source_preview(output, tmp_path / 'public')
    assert original['vertices_url'] != preview['vertices_url']
    assert 'source_format' not in original
