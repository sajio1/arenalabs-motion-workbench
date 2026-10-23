"""Resolve fixed physical humanoid semantics without robot/action policy branches.

Explicit metadata is authoritative and validated. Otherwise the four confirmed
endpoint body names constrain limb ancestry. Anatomical joint labels resolve
proximal centers; anonymous branches are accepted only when neutral geometry
has exactly the shoulder/elbow/wrist or hip/knee/ankle centers. Ambiguous rigs
must supply metadata instead of receiving a guessed solver configuration.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mujoco as mj
import numpy as np

ENDPOINTS = ('left_wrist', 'right_wrist', 'left_ankle', 'right_ankle')
SEGMENTS = tuple((side+'_'+a, side+'_'+b) for side in ('left', 'right')
                 for a, b in (('shoulder', 'elbow'), ('elbow', 'wrist'),
                              ('hip', 'knee'), ('knee', 'ankle')))
REQUIRED = ('root', 'chest') + tuple(dict.fromkeys(x for pair in SEGMENTS for x in pair))


class RobotMetadataError(ValueError):
    """The physical rig cannot be resolved unambiguously for this solver."""


def _chain(model, body):
    result = []
    while body > 0:
        result.append(int(body))
        body = int(model.body_parentid[body])
    return result[::-1]


def _body(model, name, role):
    if isinstance(name, dict):
        name = name.get('body')
    if not isinstance(name, str):
        raise RobotMetadataError(f'{role} requires an explicit native body name')
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    if bid <= 0:
        raise RobotMetadataError(f'{role} body is absent from the local robot model: {name}')
    return int(bid)


def _load(row):
    path = row.get('xml', row.get('model'))
    if not path:
        raise RobotMetadataError('Local robot xml/model path is required')
    model = mj.MjModel.from_xml_path(str(path))
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    free = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or int(model.jnt_qposadr[free[0]]) != 0:
        raise RobotMetadataError('One free root at qpos[0:7] is required')
    return model, data, int(model.jnt_bodyid[free[0]])


def _with_endpoints(row):
    """Reuse the established descriptor/tree resolver, without a neural service."""
    if row.get('endpoints'):
        return row
    row = copy.deepcopy(row)
    explicit = row.get('semantic_metadata', {}).get('semantics', {})
    if all(role in explicit for role in ENDPOINTS):
        row['endpoints'] = {role: explicit[role] for role in ENDPOINTS}
        return row
    if not row.get('body'):
        raise RobotMetadataError('Robot body descriptor name is required to resolve missing endpoints')
    from alphamotion.engine.descriptor import build_from_mjcf
    from greenwich_pipelines.robot_semantics import resolve_endpoints, ROLES
    spec, *_ = build_from_mjcf(str(row.get('xml', row.get('model'))), row['body'])
    indices = resolve_endpoints(spec)
    row['endpoints'] = {role: spec.joint_names[i] for role, i in zip(ROLES, indices)}
    return row


def _endpoint_ids(row, model):
    endpoints = row.get('endpoints', {})
    missing = [role for role in ENDPOINTS if role not in endpoints]
    if missing:
        raise RobotMetadataError('Confirmed endpoint metadata is required: '+', '.join(missing))
    result = {role: _body(model, endpoints[role], role) for role in ENDPOINTS}
    if len(set(result.values())) != 4:
        raise RobotMetadataError('The four endpoint bodies must be distinct')
    return result


def _validate(row, model, data, root):
    names = row.get('semantic_metadata', {}).get('semantics', {})
    missing = [role for role in REQUIRED if role not in names]
    if missing:
        raise RobotMetadataError('Missing physical semantic roles: '+', '.join(missing))
    ids = {role: _body(model, names[role], role) for role in REQUIRED}
    endpoints = _endpoint_ids(row, model)
    if ids['root'] != root:
        raise RobotMetadataError('Semantic root must be the actual floating root body')
    for role, bid in endpoints.items():
        if ids[role] != bid:
            raise RobotMetadataError(f'{role} semantics disagree with confirmed endpoint body')
    for role, bid in ids.items():
        if root not in _chain(model, bid):
            raise RobotMetadataError(f'{role} is not under the floating root')
    for side in ('left', 'right'):
        for base, roles in (('chest', ('shoulder', 'elbow', 'wrist')),
                            ('root', ('hip', 'knee', 'ankle'))):
            ordered = [ids[base]] + [ids[side+'_'+part] for part in roles]
            for parent, child in zip(ordered, ordered[1:]):
                if parent not in _chain(model, child):
                    raise RobotMetadataError(f'{side} {roles} do not follow physical ancestry')
            for a, b in zip(ordered[1:], ordered[2:]):
                if np.linalg.norm(data.xpos[b]-data.xpos[a]) <= 1e-6:
                    raise RobotMetadataError(f'Degenerate {side} limb segment; supply real joint centers')
    for part in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'):
        a, b = ids['left_'+part], ids['right_'+part]
        if a == b or a in _chain(model, b) or b in _chain(model, a):
            raise RobotMetadataError(f'Left/right {part} must be on separate branches')
    return ids


def validate_robot_metadata(row):
    """CPU-only preflight; returns validated semantic body IDs, no neural model."""
    row = _with_endpoints(row.get('pipeline_row', row))
    return _validate(row, *_load(row))


def _joint_label_candidates(model, branch, part):
    result = []
    for bid in branch:
        first, count = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
        labels = [(model.joint(j).name or '').lower() for j in range(first, first+count)]
        if any(part in label for label in labels):
            result.append(bid)
    return result


def _anonymous_centers(model, data, branch):
    # Fixed decoration links are not anatomical joint centers. Keep the explicit
    # endpoint even if fixed, and collapse co-located serial rotation axes.
    candidates = [b for b in branch if model.body_jntnum[b] > 0 or b == branch[-1]]
    groups = []
    for bid in candidates:
        if not groups or np.linalg.norm(data.xpos[bid]-data.xpos[groups[-1][0]]) > 1e-6:
            groups.append([bid])
        else:
            groups[-1].append(bid)
    if len(groups) != 3:
        raise RobotMetadataError('Anonymous limb has ambiguous neutral centers; provide semantic_metadata.semantics')
    return groups[0][0], groups[1][0]


def ensure_robot_metadata(row):
    """Return an independent pipeline row with validated fixed physical metadata.

    Does not alter an explicit mapping, model, palm frame, limit or solver weight.
    New rigs with ambiguous anatomy require explicit semantic body names.
    """
    row = _with_endpoints(copy.deepcopy(row.get('pipeline_row', row)))
    model, data, root = _load(row)
    existing = row.get('semantic_metadata', {}).get('semantics')
    if existing:
        _validate(row, model, data, root)
        return row
    endpoints = _endpoint_ids(row, model)
    chains = {role: _chain(model, bid) for role, bid in endpoints.items()}
    common = set(chains['left_wrist']) & set(chains['right_wrist'])
    chest = next((i for i in reversed(chains['left_wrist']) if i in common), None)
    if chest is None:
        raise RobotMetadataError('Wrist chains do not share a floating-root ancestry')
    ids = dict(root=root, chest=chest, **endpoints)
    evidence = {}
    for side in ('left', 'right'):
        for end, base, parts in (('wrist', chest, ('shoulder', 'elbow')),
                                 ('ankle', root, ('hip', 'knee'))):
            chain = chains[side+'_'+end]
            if base not in chain:
                raise RobotMetadataError(f'{side} {end} is outside the expected limb tree')
            branch = chain[chain.index(base)+1:]
            if not branch:
                raise RobotMetadataError(f'{side} {end} cannot be the branch root')
            candidates = [_joint_label_candidates(model, branch, part) for part in parts]
            if all(candidates):
                # Multi-axis shoulder/hip names label one anatomical joint group;
                # the proximal body is the established local metadata convention.
                proximal, middle = (items[0] for items in candidates)
                evidence[side+'_'+end] = 'physical joint labels along confirmed endpoint ancestry'
            elif any(candidates):
                raise RobotMetadataError(f'Incomplete {side} {end} anatomical labels; supply explicit semantics')
            else:
                proximal, middle = _anonymous_centers(model, data, branch)
                evidence[side+'_'+end] = 'exactly three distinct articulated neutral centers'
            ids[side+'_'+parts[0]] = proximal
            ids[side+'_'+parts[1]] = middle
    row['semantic_metadata'] = dict(
        semantics={role: model.body(bid).name for role, bid in ids.items()},
        semantic_body_ids=ids,
        semantic_positions_m={role: data.xpos[bid].tolist() for role, bid in ids.items()},
        metadata_provenance='Private topology/neutral-geometry resolution; no robot or action policy branches',
        resolution_evidence=evidence,
        model_sha256=hashlib.sha256(Path(row.get('xml', row.get('model'))).read_bytes()).hexdigest(),
    )
    _validate(row, model, data, root)
    return row
