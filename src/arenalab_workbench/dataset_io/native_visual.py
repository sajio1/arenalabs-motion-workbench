"""Bind displayed mesh poses to native world state using the producing model.

Build-time check only: customer models stay local. Mounted-hand render parity
is provenance evidence, not validation of a physical hand assembly.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def validate_native_visual(source, model_path, playback_path, scene, robot, output):
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from arenalab_workbench.source_display import display_basis

    source, model_path, playback_path = map(Path, (source, model_path, playback_path))
    if 'source_world_to_display' not in scene:
        raise ValueError('Native visual validation requires an explicit source_world_to_display basis')
    basis = display_basis(scene['source_world_to_display'])
    playback = json.loads(playback_path.read_text())
    frames = int(playback['frames'])
    positions = np.asarray(playback['positions'], dtype=float)
    quaternions = np.asarray(playback['quaternions'], dtype=float)
    if (frames < 1 or playback.get('fps') != 30 or positions.ndim != 3 or
            positions.shape[0] != frames or positions.shape[1] < 1 or positions.shape[2] != 3 or
            quaternions.shape != positions.shape[:2] + (4,) or not np.isfinite(positions).all() or
            not np.isfinite(quaternions).all() or
            not np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-4, rtol=0)):
        raise ValueError('Native visual playback has invalid mesh poses')
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4)).as_matrix().reshape(positions.shape[:2] + (3, 3))
    world_positions = positions @ basis
    world_rotations = basis.T @ rotations
    model = mj.MjModel.from_xml_path(str(model_path))
    with np.load(source, allow_pickle=False) as data:
        qpos = np.asarray(data['qpos_world'] if 'qpos_world' in data else data['qpos'], dtype=float)
        if 'qpos_world' in data and 'qpos' in data and not np.array_equal(data['qpos_world'], data['qpos']):
            raise ValueError('Native visual source has ambiguous qpos')
        if qpos.shape != (frames, model.nq) or not np.isfinite(qpos).all() or float(data['fps']) != 30:
            raise ValueError('Native visual source clock/state differs')
    render_path = next((p for p in (source.parent / 'robot-render.npz', source.resolve().parent / 'robot-render.npz') if p.is_file()), None)
    saved_position_error = saved_rotation_error = None
    if render_path is not None:
        with np.load(render_path, allow_pickle=False) as render:
            ids = np.asarray(render['native_body_geom_ids'])
            saved_positions = np.asarray(render['positions'], dtype=float)
            saved_rotations = np.asarray(render['rotations'], dtype=float)
        if (saved_positions.shape != world_positions.shape or saved_rotations.shape != world_rotations.shape or
                not np.isfinite(saved_positions).all() or not np.isfinite(saved_rotations).all()):
            raise ValueError('Native saved render does not cover every playback mesh and frame')
        saved_position_error = float(np.max(np.abs(saved_positions - world_positions)))
        saved_rotation_error = float(np.max(np.abs(saved_rotations - world_rotations)))
        if max(saved_position_error, saved_rotation_error) > 2e-6:
            raise ValueError('Native saved render/playback mesh poses differ')
        selection = 'retained_native_body_geom_ids'
    elif robot.get('hand') == 'robotiq-2f-140':
        # Exactly the existing gripper_export body-prefix selection. No guessed
        # mesh order, and no claim about the separately mounted hand model.
        from alphamotion.viz.kinematics import visual_mesh_geom_ids
        ids = np.asarray([g for g in visual_mesh_geom_ids(model)
                          if not any(word in (model.body(int(model.geom_bodyid[g])).name + ' ' +
                                      model.mesh(int(model.geom_dataid[g])).name).lower()
                                     for word in ('hand', 'finger', 'palm', 'dummy', 'thumb'))], dtype=int)
        selection = 'gripper_export_visual_body_prefix'
    else:
        raise ValueError('Native visual body geometry mapping is unavailable; cannot certify root correspondence')
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) < 1 or len(ids) > positions.shape[1] or
            len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= model.ngeom)):
        raise ValueError('Native visual body geometry mapping is invalid')
    state = mj.MjData(model)
    position_error = rotation_error = 0.
    for index, row in enumerate(qpos):
        state.qpos[:] = row
        mj.mj_forward(model, state)
        position_error = max(position_error, float(np.max(np.abs(state.geom_xpos[ids] - world_positions[index, :len(ids)]))))
        rotation_error = max(rotation_error, float(np.max(np.abs(state.geom_xmat[ids].reshape(-1, 3, 3) - world_rotations[index, :len(ids)]))))
    if max(position_error, rotation_error) > 2e-6:
        raise ValueError('Native body FK/playback differs: root, model or geometry prefix is inconsistent')
    receipt = dict(schema='arenalabs-native-visual-validation-v1', passed=True, frames=frames,
                   robot_id=robot['id'], body_meshes=len(ids), total_meshes=positions.shape[1],
                   checked_body_mesh_poses=frames * len(ids), geometry_selection=selection,
                   source_world_to_display=basis.tolist(), source_sha256=sha(source),
                   model_source_sha256=sha(model_path), playback_sha256=sha(playback_path),
                   body_position_max_m=position_error, body_rotation_matrix_max=rotation_error,
                   saved_render_all_meshes_checked=render_path is not None,
                   saved_render_position_max_m=saved_position_error,
                   saved_render_rotation_matrix_max=saved_rotation_error,
                   attachment_fresh_fk_validated=False, dynamics_validated=False)
    target = Path(output) / 'visual-validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt
