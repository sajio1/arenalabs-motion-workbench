"""Fresh native FK must agree with the displayed floating root and mesh order."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

mj = pytest.importorskip('mujoco')
from arenalab_workbench.dataset_io.native_visual import validate_native_visual

BASIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
XML = '''<mujoco><asset><mesh name="body_visual" vertex="-.1 -.1 -.1  .1 -.1 -.1  -.1 .1 -.1  .1 .1 -.1  -.1 -.1 .1  .1 -.1 .1  -.1 .1 .1  .1 .1 .1"/></asset>
<worldbody><body name="base" pos="0 0 1"><freejoint name="root"/>
<geom type="mesh" mesh="body_visual" contype="0" conaffinity="0"/>
<body name="arm" pos=".2 0 .1"><joint name="hinge" axis="0 1 0"/>
<geom type="mesh" mesh="body_visual" pos=".1 0 0" contype="0" conaffinity="0"/>
</body></body></worldbody></mujoco>'''


@pytest.fixture
def case(tmp_path):
    model_path = tmp_path / 'private-robot.xml'
    model_path.write_text(XML)
    model = mj.MjModel.from_xml_path(str(model_path))
    qpos = np.tile(model.qpos0, (3, 1))
    qpos[:, 0] = [0, .1, .3]
    qpos[:, 7] = [0, .2, .5]
    source = tmp_path / 'motion.npz'
    np.savez(source, qpos=qpos, fps=30)
    state = mj.MjData(model)
    positions, rotations = [], []
    for row in qpos:
        state.qpos[:] = row
        mj.mj_forward(model, state)
        positions.append(state.geom_xpos.copy())
        rotations.append(state.geom_xmat.copy().reshape(-1, 3, 3))
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    # An optional attachment render channel is retained but not claimed to have
    # fresh FK coverage from the body-only producing model.
    positions = np.concatenate([positions, positions[:, -1:] + [.05, 0, 0]], axis=1)
    rotations = np.concatenate([rotations, rotations[:, -1:]], axis=1)
    render = tmp_path / 'robot-render.npz'
    np.savez(render, positions=positions, rotations=rotations, native_body_geom_ids=[0, 1])
    playback = dict(frames=3, fps=30, positions=(positions @ BASIS.T).tolist(),
                    quaternions=Rotation.from_matrix((BASIS @ rotations).reshape(-1, 3, 3)).as_quat().reshape(3, 3, 4).tolist())
    playback_path = tmp_path / 'playback.json'
    playback_path.write_text(json.dumps(playback))
    return SimpleNamespace(source=source, qpos=qpos, model=model_path, render=render,
                           positions=positions, rotations=rotations, playback=playback,
                           playback_path=playback_path, scene={'source_world_to_display': BASIS.tolist()},
                           robot={'id': 'g1', 'hand': 'wuji'}, output=tmp_path / 'native')


def validate(case):
    return validate_native_visual(case.source, case.model, case.playback_path,
                                  case.scene, case.robot, case.output)


def test_native_body_and_saved_attachment_render_are_separately_certified(case):
    result = validate(case)
    assert result['passed'] and result['checked_body_mesh_poses'] == 6
    assert result['total_meshes'] == 3 and result['saved_render_all_meshes_checked']
    assert not result['attachment_fresh_fk_validated'] and not result['dynamics_validated']
    assert str(case.source.parent) not in json.dumps(result)


@pytest.mark.parametrize('failure', ['root_position', 'root_orientation', 'model_offset', 'prefix_order'])
def test_identical_joint_angles_do_not_hide_wrong_root_model_or_mesh_order(case, failure):
    if failure == 'root_position':
        case.qpos[:, 0] += .02
        np.savez(case.source, qpos=case.qpos, fps=30)
    elif failure == 'root_orientation':
        case.qpos[:, 3:7] = [np.cos(.1), 0, 0, np.sin(.1)]
        np.savez(case.source, qpos=case.qpos, fps=30)
    elif failure == 'model_offset':
        case.model.write_text(XML.replace('pos=".2 0 .1"', 'pos=".3 0 .1"'))
    else:
        np.savez(case.render, positions=case.positions, rotations=case.rotations, native_body_geom_ids=[1, 0])
    with pytest.raises(ValueError, match='body FK/playback differs'):
        validate(case)
    assert not (case.output / 'visual-validation.json').exists()


def test_stored_attachment_pose_drift_is_detected_without_a_hand_model(case):
    case.playback['positions'][1][2][0] += .01
    case.playback_path.write_text(json.dumps(case.playback))
    with pytest.raises(ValueError, match='saved render/playback'):
        validate(case)


@pytest.mark.parametrize('basis', [None, [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]])
def test_declared_display_basis_cannot_be_omitted_or_guessed(case, basis):
    case.scene = {} if basis is None else {'source_world_to_display': basis}
    with pytest.raises(ValueError):
        validate(case)


def test_unknown_mesh_mapping_is_not_certified(case):
    case.render.unlink()
    with pytest.raises(ValueError, match='geometry mapping is unavailable'):
        validate(case)


def test_gripper_body_prefix_can_be_certified_without_hand_assembly(case):
    case.render.unlink()
    case.robot['hand'] = 'robotiq-2f-140'
    result = validate(case)
    assert result['passed'] and result['body_meshes'] == 2
    assert result['geometry_selection'] == 'gripper_export_visual_body_prefix'
    assert not result['saved_render_all_meshes_checked']
    assert result['saved_render_position_max_m'] is None
    assert not result['attachment_fresh_fk_validated']
