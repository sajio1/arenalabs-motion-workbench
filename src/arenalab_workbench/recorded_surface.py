"""Recorded skeleton authority with an explicitly approximate skinned surface.

The surface construction follows the documented HOI-Retarget source_body_mesh
method (ETH Zurich, MIT, b341dadde3fc745ef5ea606eaacbde9a0cbb916f).
This is a distinct source contract: it never relaxes the SMPL decoded-FK check,
fits a new body, or claims that the proxy girth is the subject's original shape.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _source(raw, spec, take, parents, rest, rotation):
    from .motion_sources import _finite
    if spec.get('family') != 'smplx' or spec.get('skeleton_alignment') != 'model_pelvis' or spec.get('surface_method') != 'skinning_joint_offsets_v1':
        raise ValueError('Recorded skeleton requires explicit SMPL-X topology, model_pelvis alignment and skinning_joint_offsets_v1 surface')
    source = np.asarray(raw['source_joints'], dtype=np.float64)[take]
    mask = np.asarray(raw['source_joints_mask'])[take]
    if source.shape != (len(take), len(parents), 3) or mask.shape != source.shape[:2] or mask.dtype != np.bool_:
        raise ValueError('Recorded skeleton shape/mask must match the complete model topology')
    if not mask[:, :22].all() or not np.isfinite(source[mask]).all() or not np.array_equal(mask, np.isfinite(source).all(-1)):
        raise ValueError('Recorded body22 must be finite; missing extra joints require an explicit mask')
    translation = _finite(raw['source_to_model_translation_m'], 'source_to_model_translation_m')[take]
    if translation.shape != (len(take), 3):
        raise ValueError('Recorded skeleton alignment translation must match source frames')
    source = source + translation[:, None]
    adjusted = rest.copy()
    for j in range(1, len(parents)):
        valid = mask[:, j] & mask[:, parents[j]]
        if valid.any():
            offset = np.einsum('tji,tj->ti', rotation[valid, parents[j]], source[valid, j] - source[valid, parents[j]])
            adjusted[j] = adjusted[parents[j]] + np.median(offset, axis=0)
        else:
            adjusted[j] = adjusted[parents[j]] + rest[j] - rest[parents[j]]
    return source, mask, translation, adjusted


def _weights_along(shaped, rest, parents, weights):
    child = np.arange(len(parents))
    for j in range(1, 22):
        children = np.flatnonzero(parents[1:22] == j) + 1
        if len(children) == 1:
            child[j] = children[0]
    bone = rest[child] - rest
    along = np.einsum('vjc,jc->vj', shaped[:, None] - rest, bone) / np.maximum(np.sum(bone ** 2, axis=1), 1e-9)
    return child, weights * np.clip(along, 0., 1.)


def _neutral_proxy(shaped, rest, adjusted, parents, weights):
    child, along = _weights_along(shaped, rest, parents, weights)
    delta = adjusted - rest
    return shaped + weights @ delta + along @ (delta[child] - delta)


def decode_recorded_surface(raw, spec, take, model_path):
    from .motion_sources import _decode
    if float(np.asarray(raw.get('body_scale', spec.get('body_scale', 1.)))) != 1.:
        raise ValueError('Recorded skeleton positions are already in metres; separate model scale is unsupported')
    base_spec = dict(spec, format='smpl_parameters')
    # fitted_joints is meaningful only for an actual fitted SMPL model. A
    # recorded skeleton is validated below, independently of the proxy model.
    base = _decode({k: v for k, v in raw.items() if k != 'fitted_joints'}, base_spec, take, model_path)
    joints, rotation, vertices, shaped, rest, parents, weights, faces, beta, check = base
    source, mask, translation, adjusted = _source(raw, spec, take, parents, rest, rotation)
    if not np.allclose(source[:, 0], joints[:, 0], atol=2e-6, rtol=0):
        raise ValueError('Recorded pelvis alignment disagrees with the provided SMPL orientation model')
    delta = np.zeros_like(joints)
    delta[:, :22] = source[:, :22] - joints[:, :22]
    for j in range(22, len(parents)):
        delta[:, j] = delta[:, parents[j]]
    child, along = _weights_along(shaped, rest, parents, weights)
    proxy = vertices + np.einsum('vj,tjc->tvc', weights, delta) + np.einsum('vj,tjc->tvc', along, delta[:, child] - delta)
    for wrist, first in ((20, 25), (21, 40)):
        chain = [wrist, first + 3, first + 4, first + 5]
        valid = mask[:, chain].all(axis=1)
        if not valid.any():
            continue
        actual = np.median(np.linalg.norm(np.diff(source[valid][:, chain], axis=1), axis=-1).sum(1))
        baseline = np.median(np.linalg.norm(np.diff(joints[valid][:, chain], axis=1), axis=-1).sum(1))
        if baseline <= 1e-8 or actual <= 1e-8:
            raise ValueError('Degenerate recorded hand anatomy')
        pivot = (joints[:, wrist] + delta[:, wrist])[:, None]
        influence = weights[:, [wrist, *range(first, first + 15)]].sum(1)
        proxy += (actual / baseline - 1.) * influence[None, :, None] * (proxy - pivot)
    authoritative = np.where(mask[..., None], source, joints + delta)
    check.update(source_geometry='recorded skeleton; exact body22 positions',
                 source_surface='skinning_joint_offsets_v1; approximate girth, not fitted subject surface',
                 source_skeleton_alignment='one explicit per-frame translation to orientation-model pelvis',
                 source_body22_position_error_m=float(np.max(np.abs(authoritative[:, :22] - source[:, :22]))),
                 missing_extra_joint_slots=np.flatnonzero(~mask.all(axis=0)).tolist(),
                 source_alignment_max_m=float(np.linalg.norm(translation, axis=-1).max()))
    return authoritative, rotation, proxy, _neutral_proxy(shaped, rest, adjusted, parents, weights), adjusted, parents, weights, faces, beta, check


def describe_recorded_surface(raw, spec, take, model_path):
    from .motion_sources import _model, _beta, _finite
    _, shaped, rest, parents, weights, _ = _model(model_path, spec['family'], _beta(raw))
    pose = _finite(raw['poses'], 'poses').reshape(len(raw['poses']), len(parents), 3)[take]
    rotation = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(take), len(parents), 3, 3)
    for j in range(1, len(parents)):
        rotation[:, j] = rotation[:, parents[j]] @ rotation[:, j]
    _, _, _, adjusted = _source(raw, spec, take, parents, rest, rotation)
    return _neutral_proxy(shaped, rest, adjusted, parents, weights), adjusted
