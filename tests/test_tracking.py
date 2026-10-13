import numpy as np
from scipy.spatial.transform import Rotation
from greenwich_pipelines.viewer.tracking import wrist_metrics
from greenwich_pipelines.viewer.export import join_playback_segments


def test_wrist_errors_have_mm_units_and_geodesic_angles():
    goal_p = np.zeros((2, 2, 3))
    actual_p = goal_p.copy()
    actual_p[:, 0] = [.003, .004, 0]
    goal_r = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
    actual_r = goal_r.copy()
    actual_r[:, 0] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    actual_r[:, 1] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    tracking = wrist_metrics(actual_p, goal_p, actual_r, goal_r)
    assert np.allclose(tracking['position_error_mm'], [[5, 0], [5, 0]])
    assert np.allclose(tracking['orientation_error_deg'], [[90, 180], [90, 180]])
    # A common coordinate rotation / translation must leave tracking errors unchanged.
    basis = Rotation.from_euler('xyz', [23, 50, -41], degrees=True).as_matrix()
    changed = wrist_metrics(actual_p @ basis.T + 3, goal_p @ basis.T + 3, basis @ actual_r, basis @ goal_r)
    assert np.allclose(changed['position_error_mm'], tracking['position_error_mm'])
    assert np.allclose(changed['orientation_error_deg'], tracking['orientation_error_deg'])


def test_segment_telemetry_preserves_each_frame_without_smoothing():
    def segment(mm):
        p = np.zeros((2, 2, 3));p[:, 0, 0] = mm / 1000
        r = np.broadcast_to(np.eye(3), (2, 2, 3, 3))
        return dict(fps=30, frames=2, geom_ids=[], parents=[], unit='m', axis='y-up',
                    positions=[[], []], quaternions=[[], []], skeleton=[[], []], targets=[[], []],
                    tracking=wrist_metrics(p, p*0, r, r))
    result = join_playback_segments([segment(1), segment(25)])
    assert result['window_boundaries'] == [0, 2]
    assert np.array_equal(np.asarray(result['tracking']['position_error_mm'])[:, 0], [1, 1, 25, 25])


def test_native_gento_tracking_uses_rotated_tcp_offset(tmp_path):
    import json
    import mujoco
    from greenwich_pipelines.viewer.tracking import tracking_payload
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="base"><freejoint/><geom size=".1"/><body name="left" pos="1 0 0" euler="0 0 90"><geom size=".1"/></body><body name="right" pos="-1 0 0"><geom size=".1"/></body></body></worldbody></mujoco>''')
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps(dict(wrists=['left', 'right'], end_effector_frames=[dict(link=name, rotation=np.eye(3).tolist()) for name in ('left', 'right')])))
    qpos = model.qpos0[None].copy()
    target = np.array([[[1, .1, 0], [-.9, 0, 0]]])
    rotations = np.array([[Rotation.from_euler('z', 90, degrees=True).as_matrix(), np.eye(3)]])
    saved = dict(qpos=qpos, targets=target, target_orientation=rotations, ee_local_position=np.array([[.1, 0, 0], [.1, 0, 0]]))
    result = tracking_payload(dict(metadata=str(metadata)), saved, 'gento', model)
    assert np.max(result['position_error_mm']) < 1e-8
    assert np.max(result['orientation_error_deg']) < 1e-5
