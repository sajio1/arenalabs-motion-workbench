import numpy as np
import pytest
from tests.workbench.test_motion_sources import fixture
from arenalab_workbench import motion_sources as m


def recorded(root):
    spec, _ = fixture(root)
    raw = m._npz(root / 'motion.npz')
    take = np.arange(len(raw['poses']))
    base = m._decode(raw, spec, take, root / 'model.npz')
    joints, rotation, _, _, rest, parents = base[:6]
    actual = joints.copy()
    for j in range(1, len(parents)):
        actual[:, j] = actual[:, parents[j]] + np.einsum('tij,j->ti', rotation[:, parents[j]], (rest[j] - rest[parents[j]]) * .72)
    shift = np.broadcast_to([.4, -.2, .1], (len(take), 3)).copy()
    raw.update(source_joints=actual - shift[:, None], source_joints_mask=np.ones(actual.shape[:2], dtype=bool), source_to_model_translation_m=shift)
    spec.update(format='recorded_skeleton_surface', skeleton_alignment='model_pelvis', surface_method='skinning_joint_offsets_v1')
    return spec, raw, take, base, actual


def test_recorded_skeleton_has_authority_and_proxy_is_separate(tmp_path):
    spec, raw, take, base, actual = recorded(tmp_path)
    result = m._decode(raw, spec, take, tmp_path / 'model.npz')
    np.testing.assert_allclose(result[0], actual, atol=1e-12)
    np.testing.assert_array_equal(result[1], base[1])
    assert not np.allclose(result[2], base[2])
    np.testing.assert_allclose(np.linalg.norm(np.diff(result[4][[1, 4]], axis=0), axis=-1),
                               .72 * np.linalg.norm(np.diff(base[4][[1, 4]], axis=0), axis=-1))
    assert result[-1]['source_body22_position_error_m'] == 0.
    np.savez(tmp_path / 'motion.npz', **raw)
    out = m.prepare_supplied_source(spec, tmp_path / 'out', model_path=tmp_path / 'model.npz')
    with np.load(out / 'native_source.npz') as d:
        np.testing.assert_allclose(d['full_body_position_yup_m'], actual[::3, :22])
    # The strict actual-SMPL contract must still reject this changed skeleton.
    strict = dict(spec, format='smpl_decoded')
    strict_raw = dict(raw, joints=result[0], global_rotation=result[1], vertices=result[2])
    with pytest.raises(ValueError, match='FK disagree'):
        m._decode(strict_raw, strict, take, tmp_path / 'model.npz')


def test_missing_face_slots_are_explicit_but_body22_cannot_be_missing(tmp_path):
    spec, raw, take, _, actual = recorded(tmp_path)
    raw['source_joints'][:, 22:25] = np.nan
    raw['source_joints_mask'][:, 22:25] = False
    result = m._decode(raw, spec, take, tmp_path / 'model.npz')
    np.testing.assert_allclose(result[0][:, :22], actual[:, :22])
    assert np.isfinite(result[2]).all()
    assert result[-1]['missing_extra_joint_slots'] == [22, 23, 24]
    raw['source_joints_mask'][:, 20] = False
    with pytest.raises(ValueError, match='body22'):
        m._decode(raw, spec, take, tmp_path / 'model.npz')


def test_wrong_or_double_applied_pelvis_alignment_fails(tmp_path):
    spec, raw, take, _, _ = recorded(tmp_path)
    raw['source_joints'] += raw['source_to_model_translation_m'][:, None]
    with pytest.raises(ValueError, match='pelvis alignment'):
        m._decode(raw, spec, take, tmp_path / 'model.npz')
