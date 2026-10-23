"""Read normalized complete source bodies. Sparse observations are resolved upstream.

Canonical Y-up row points convert to native Z-up by p @ C. Rotations used for
transfer express both local and world axes in the declared canonical basis;
physical palm frames are separately observed/calibrated and never inferred
from a six-point rotation. Every supplied body joint is visible to the codec.
"""
from pathlib import Path
import json
import numpy as np
from .prior import C,sha,_arrays,_unit,palm_frame

POLICY='supplied-full-body-all-visible-no-source-decode-v1'
ROLE_NAMES=('Hips','Head','LeftHand','RightHand','LeftFoot','RightFoot')
BONE_NAMES=tuple((side+a,side+b) for side in ('Left','Right')
                for a,b in (('Arm','ForeArm'),('ForeArm','Hand'),('Leg','Shin'),('Shin','Foot')))


def source_directory(path):
    path=Path(path).expanduser().resolve()
    if path.is_file():path=path.parent
    if (path/'native_source.npz').is_file():return path
    manifest_path=path/'input_manifest.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text())
        original=manifest.get('staged_original_source')
        if original and (Path(original)/'native_source.npz').is_file():return Path(original).resolve()
    raise FileNotFoundError('Complete normalized native_source.npz required. Supply full canonical/SMPL/H/X through the source adapter; missing body must be reconstructed upstream, never silently six-point-decoded here.')


def load_source(path):
    from alphamotion.engine.nets.rotations import SkeletonSpec
    from alphamotion.embodiment.dof_features import ball_dof
    raw=source_directory(path);manifest=json.loads((raw/'input_manifest.json').read_text())
    native=raw/'native_source.npz';digest=sha(native)
    expected=manifest.get('native_source_sha256')
    if expected and expected!=digest:raise ValueError('Immutable normalized source hash changed')
    z=_arrays(native);names=[str(x) for x in z['names']];J=len(names)
    p=np.asarray(z['full_body_position_yup_m'],np.float64);R=np.asarray(z['full_body_rotation_yup'],np.float64)
    times=np.asarray(z['timestamps_s'],np.float64);T=len(times);fps=float(z.get('exact_fps',30.))
    if T<3 or p.shape!=(T,J,3) or R.shape!=(T,J,3,3):raise ValueError('Expected complete source positions and rotations for all frames/joints')
    if abs(fps/30-1)>2e-5 or not np.allclose(np.diff(times),1/fps,rtol=0,atol=1e-8):raise ValueError('Only contiguous nominal30Hz source clocks are supported')
    if not all(np.isfinite(x).all() for x in (p,R,times)):raise ValueError('Nonfinite source body')
    if not np.allclose(R.swapaxes(-1,-2)@R,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(R),1,atol=1e-4):raise ValueError('Source rotations must be proper orthonormal matrices')
    roles=np.asarray(z['role_indices'],np.int64) if 'role_indices' in z else np.array([names.index(n) for n in ROLE_NAMES],np.int64)
    if roles.shape!=(6,) or np.any(roles<0) or np.any(roles>=J):raise ValueError('Invalid source task role indices')
    if roles[0]!=0:raise ValueError('Canonical pelvis/root must be the first body joint')
    parents=np.asarray(z['parents'],np.int64);rest=np.asarray(z['rest_offsets_cm'],np.float32)
    if parents.shape!=(J,) or rest.shape!=(J,3) or parents[0]>=0 or any(parents[j]<0 or parents[j]>=j for j in range(1,J)):raise ValueError('Expected a topologically ordered single-root source tree')
    if not np.isfinite(rest).all():raise ValueError('Nonfinite source rest geometry')
    dof=z.get('dof')
    if dof is None:dof,_=ball_dof(J)
    spec=SkeletonSpec('source_fullbody_'+digest[:16],parents,rest,names)
    world=p@C;world_R=C.T@R@C
    bones=np.asarray(z['bone_directions_world'],np.float64) if 'bone_directions_world' in z else np.stack([_unit(world[:,names.index(b)]-world[:,names.index(a)]) for a,b in BONE_NAMES],1)
    if bones.shape!=(T,8,3) or not np.isfinite(bones).all():raise ValueError('Invalid complete source limb directions')
    # Preserve saved numerical arrays exactly when the normalized package includes them.
    world=np.asarray(z.get('source_body_position_world_m',world),np.float64)
    world_R=np.asarray(z.get('source_body_rotation_world_transfer',world_R),np.float64)
    bones=np.asarray(z.get('bone_directions_world',bones),np.float64)
    six=np.asarray(z.get('six_position_zup_m',world[:,roles]),np.float64)
    reference_R=np.asarray(z.get('reference_world_R_zup',world_R[:,roles]),np.float64)
    wrists=np.asarray(z['wrist_position_world_m'],np.float64)
    sole=np.asarray(z['sole_low_z_m'],np.float64);floor=float(z['floor_z_m'])
    if six.shape!=(T,6,3) or reference_R.shape!=(T,6,3,3) or wrists.shape!=(T,2,3) or sole.shape!=(T,2):raise ValueError('Source endpoint/sole shapes disagree')
    if not all(np.isfinite(x).all() for x in (six,reference_R,wrists,sole,np.asarray(floor))):raise ValueError('Nonfinite task references')
    if not np.allclose(six[:,2:4],wrists,atol=1e-8):raise ValueError('Wrist observations disagree with six task view')
    hashes={p.name:sha(p) for p in raw.iterdir() if p.is_file() and p.suffix in ('.json','.npz')}
    return dict(source_dir=str(raw),raw_source_dir=str(raw),frames=T,n=T,fps=30.,exact_fps=fps,
        floor=floor,floor_z_m=floor,times=times,source_spec=spec,spec=spec,source_dof=np.asarray(dof),
        names=names,parents=parents,roles=roles,source_hashes=hashes,source_format=manifest.get('schema','arenalabs.source-body.v1'),
        morphology_scale=manifest.get('morphology_scale'),source_body_provenance=manifest.get('source_body',manifest.get('source_body_provenance','Supplied complete body; provenance in input manifest')),
        position_yup_m=six@C.T,rotation_yup=C@reference_R@C.T,reference_world_pos_zup=six,
        reference_world_R_zup=reference_R,six_position_zup_m=six,sole_low_z_m=sole,
        obs_wrist_zup_m=wrists,palm_R_native=palm_frame(z['palm_forward'],z['palm_normal']),
        inferred=False,inferred_world_pos_zup=world,inferred_world_R_zup=world_R,
        inferred_bone_directions=bones,full_body_position_yup_m=p,full_body_rotation_yup=R,
        source_frame_indices=z.get('source_frame_indices',np.arange(T)),source_decoder_used=False,
        source_body_modified=bool(manifest.get('source_uniform_similarity_applied')),C_native_to_yup=C.copy())
