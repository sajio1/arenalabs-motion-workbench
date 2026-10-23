"""Local, pose-independent uniform morphology scaling of prepared full-body sources.

This is a similarity of the supplied human body and scene, not a new SMPL fit or
changed betas. No network, neural inference, optimization, or GPU is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

import mujoco
import numpy as np

POLICY = 'eight-limb-segment-uniform-least-squares-ground-anchor-v1'
C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
SOURCE_ROLES = dict(left_shoulder='LeftArm', right_shoulder='RightArm',
                    left_elbow='LeftForeArm', right_elbow='RightForeArm',
                    left_wrist='LeftHand', right_wrist='RightHand',
                    left_hip='LeftLeg', right_hip='RightLeg',
                    left_knee='LeftShin', right_knee='RightShin',
                    left_ankle='LeftFoot', right_ankle='RightFoot')
SEGMENTS = tuple((side+'_'+a, side+'_'+b) for side in ('left', 'right')
                 for a, b in (('shoulder', 'elbow'), ('elbow', 'wrist'),
                              ('hip', 'knee'), ('knee', 'ankle')))
NPZ_FILES = ('native_source.npz', 'canonical.npz', 'observations.npz',
             'source_mesh.npz', 'smplx_motion.npz', 'objects.npz')
MANIFESTS = ('input_manifest.json', 'source_provenance.json', 'provenance.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fit_scale(row, source):
    """Fit s = argmin_s sum_i (s * human_length_i - robot_length_i)^2.

    ``source`` is a prepared native_source array dictionary or load_source dict.
    Robot lengths are neutral MJCF semantic-center distances, not task poses.
    Human median bone lengths cannot collapse when the actor bends or crouches.
    All eight segments use the same policy and equal absolute-length weight.
    """
    row = row.get('pipeline_row', row)
    robot_source = 'semantic_position_world_m' in source
    names = [str(x) for x in source['semantic_names' if robot_source else 'names']]
    P = source.get('source_body_position_world_m', source.get('inferred_world_pos_zup'))
    if P is None and 'full_body_position_yup_m' in source:
        P = np.asarray(source['full_body_position_yup_m']) @ C
    if P is None:
        raise ValueError('Source must contain the supplied full-body world positions')
    if robot_source:P = source['semantic_position_world_m']
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 3 or P.shape[1:] != (len(names), 3) or not np.isfinite(P).all():
        raise ValueError('Expected finite [T,J,3] supplied body positions')
    model_path = Path(row.get('xml', row.get('model'))).resolve()
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_kinematics(m, d)
    semantics = row['semantic_metadata']['semantics']
    lengths = []
    for a, b in SEGMENTS:
        source_a, source_b = names.index(a if robot_source else SOURCE_ROLES[a]), names.index(b if robot_source else SOURCE_ROLES[b])
        h = np.linalg.norm(P[:, source_b] - P[:, source_a], axis=-1)
        r = float(np.linalg.norm(d.xpos[m.body(semantics[b]).id] - d.xpos[m.body(semantics[a]).id]))
        human = float(np.median(h))
        if min(human, r) <= 1e-8:
            raise ValueError(f'Degenerate physical segment: {a} to {b}')
        lengths.append(dict(start=a, end=b, human_length_m=human, robot_length_m=r,
                            human_observed_min_m=float(h.min()), human_observed_max_m=float(h.max()),
                            robot_over_human=r/human))
    h = np.array([x['human_length_m'] for x in lengths])
    r = np.array([x['robot_length_m'] for x in lengths])
    factor = float(h @ r / (h @ h))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Invalid uniform similarity factor')
    floor = float(source.get('floor_z_m', source.get('floor')))
    root_index = names.index('root' if robot_source else 'Hips')
    anchor = np.r_[P[0, root_index, :2], floor]
    for item in lengths:
        item['scaled_human_length_m'] = factor * item['human_length_m']
        item['remaining_robot_minus_scaled_human_m'] = item['robot_length_m'] - item['scaled_human_length_m']
    return dict(policy=POLICY, scale=factor, factor=factor, robot_id=row['name'],
                anchor_world_m=anchor.tolist(), anchor_source_yup_m=(anchor @ C.T).tolist(),
                formula='p_scaled = anchor + scale * (p_original - anchor); local vectors *= scale',
                scale_fit='sum(h_i*r_i) / sum(h_i*h_i), eight equally weighted absolute-length residuals',
                robot_length_basis='Exact local MJCF qpos0 semantic joint centers; two arm and two leg segments per side',
                human_length_basis='Median original supplied body segment lengths across selected frames, not compressed posed height',
                world_basis_source_to_native=C.T.tolist(), segments=lengths,
                length_fit_rms_m=float(np.sqrt(np.mean((factor*h-r)**2))),
                robot_xml=str(model_path), robot_xml_sha256=sha(model_path),
                robot_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                module_sha=sha(__file__), interaction_scope='Normalized interaction: the human and all scene object sizes/translations share one similarity; not the original fixed-size object task',
                betas_changed=False, rotations_changed=False, poses_changed=False,
                timestamps_changed=False, source_redecoded=False,
                morphology_limit='One scalar cannot remove differing arm/leg/torso proportions or certify reachability')


def world_points(value, scaling, *, basis='native', centimetres=False):
    """Apply the same anchored similarity in either declared world basis."""
    if basis not in ('native', 'source'):
        raise ValueError('Unknown world basis')
    anchor = np.asarray(scaling['anchor_world_m' if basis == 'native' else 'anchor_source_yup_m'])
    if centimetres:
        anchor = anchor * 100.
    return anchor + scaling['scale'] * (np.asarray(value, dtype=np.float64) - anchor)


def world_height(value, scaling):
    anchor_z = scaling['anchor_world_m'][2]
    return anchor_z + scaling['scale'] * (np.asarray(value, dtype=np.float64) - anchor_z)


def transform_arrays(filename, original, scaling):
    """Explicit schema transforms; all unlisted pose/rotation/time arrays copy unchanged."""
    out = {k: np.array(v, copy=True) for k, v in original.items()}
    factor = scaling['scale']
    native_fields = {
        'native_source.npz': ('source_body_position_world_m', 'six_position_zup_m', 'wrist_position_world_m', 'semantic_position_world_m'),
        'source_mesh.npz': ('vertices_world_m', 'wrist_position_world_m', 'pelvis_position_world_m'),
        'observations.npz': ('wrist_position', 'palm_position', 'head_position'),
        'smplx_motion.npz': ('joints',),
        'objects.npz': ('positions_world_m',),
    }
    source_fields = {'native_source.npz': ('full_body_position_yup_m',),
                     'smplx_motion.npz': ('trans', 'joints_source_m')}
    height_fields = {'native_source.npz': ('sole_low_z_m', 'floor_z_m'),
                     'source_mesh.npz': ('floor_world_m',),
                     'observations.npz': ('floor_z',),
                     'smplx_motion.npz': ('sole_low_z_m',)}
    for key in native_fields.get(filename, ()):
        if key not in original:continue
        out[key] = world_points(original[key], scaling)
        if key == 'vertices_world_m':
            out[key] = out[key].astype(original[key].dtype)
    for key in source_fields.get(filename, ()):
        if key in original:out[key] = world_points(original[key], scaling, basis='source')
    for key in height_fields.get(filename, ()):
        if key in original:out[key] = world_height(original[key], scaling)
    if filename in ('canonical.npz', 'native_source.npz'):
        out['rest_offsets_cm'] = original['rest_offsets_cm'] * factor
    if filename == 'canonical.npz':
        out['world_position_cm'] = world_points(original['world_position_cm'], scaling, basis='source', centimetres=True)
        metadata = json.loads(str(original.get('metadata', '{}')))
        metadata.update(morphology_scale=scaling, source_uniform_similarity_applied=True,
                        source_pose_or_translation_modified=True, source_pose_rotations_modified=False,
                        betas_modified=False, interaction_scope=scaling['interaction_scope'])
        out['metadata'] = np.array(json.dumps(metadata, sort_keys=True))
    if filename == 'smplx_motion.npz':
        out['rest_joints_source_m'] = original['rest_joints_source_m'] * factor
        out['trans_original_m'] = np.array(original['trans'], copy=True)
        out['trans_original'] = np.array(original['trans'], copy=True)
        out['uniform_scale'] = np.array(factor)
        out['similarity_anchor_world_m'] = np.array(scaling['anchor_world_m'])
        out['similarity_anchor_source_yup_m'] = np.array(scaling['anchor_source_yup_m'])
        out['uniform_scale_semantics'] = np.array('Scale local SMPL-X shaped/posed geometry and rest joints by uniform_scale, then use transformed trans; betas and poses unchanged. Trans alone is insufficient with the original unscaled SMPL model.')
    if filename == 'objects.npz':
        for key in original:
            if re.fullmatch(r'object_\d+_vertices_local_m', key):
                out[key] = original[key] * factor
    return out


def materialize_source(source_dir,row,outdir):
    """Apply the validated similarity to normalized body/mesh/object fields only.

    Original licensed model parameters remain in the original provenance. A
    scaled mesh must not be mislabeled as an unscaled SMPL model plus new trans.
    """
    from .sources import source_directory
    source_dir=source_directory(source_dir);outdir=Path(outdir).resolve()
    if outdir==source_dir or source_dir in outdir.parents:raise ValueError('Scaled output must be outside immutable source')
    if outdir.exists():raise FileExistsError('Scaled source output already exists: '+str(outdir))
    manifest=json.loads((source_dir/'input_manifest.json').read_text())
    if manifest.get('source_uniform_similarity_applied') or manifest.get('morphology_scale'):
        raise ValueError('Refusing to scale an already transformed source twice')
    original=arrays(source_dir/'native_source.npz')
    scaling=fit_scale(row,original)
    inputs={p.name:sha(p) for p in source_dir.iterdir() if p.is_file() and p.suffix in ('.npz','.json')}
    scaling.update(original_source_dir=str(source_dir),original_source_sha256=inputs)
    outdir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=outdir.name+'.prepare-',dir=outdir.parent) as temporary:
        temp=Path(temporary)
        for name in ('native_source.npz','source_mesh.npz','canonical.npz','observations.npz','objects.npz'):
            path=source_dir/name
            if path.is_file():np.savez_compressed(temp/name,**transform_arrays(name,arrays(path),scaling))
        target=source_dir/'operation-accuracy/source-targets.npz'
        if target.is_file():
            arr=arrays(target);arr['points']=world_points(arr['points'],scaling)
            (temp/'operation-accuracy').mkdir();np.savez_compressed(temp/'operation-accuracy/source-targets.npz',**arr)
        output_hashes={p.name:sha(p) for p in temp.glob('*.npz')}
        final=dict(manifest,schema='arenalabs.normalized-source-body.v1',
                   native_source_sha256=output_hashes['native_source.npz'],output_sha256=output_hashes,
                   morphology_scale=scaling,source_uniform_similarity_applied=True,
                   original_source_dir=str(source_dir),original_source_sha256=inputs,
                   original_source_body=manifest.get('source_body'),source_body_modified=True,
                   interaction_scope=scaling['interaction_scope'],source_pose_rotations_modified=False,
                   source_translation_modified=True,betas_modified=False,timestamps_modified=False)
        for name in ('input_manifest.json','source_provenance.json'):
            (temp/name).write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
        if {name:sha(source_dir/name) for name in inputs}!=inputs:raise RuntimeError('Original source changed while scaling')
        temp.rename(outdir)
    return final
