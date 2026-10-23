import json
from pathlib import Path
import numpy as np
import pytest
mj=pytest.importorskip('mujoco')
pytest.importorskip('torch')
pytest.importorskip('trimesh')
from arenalab_workbench.gripper_export import native_motion_state
from arenalab_workbench.gripper_retarget import source_human_metadata


def test_native_qpos_supplies_world_pose_and_scalar_values(tmp_path):
    model=mj.MjModel.from_xml_string('<mujoco><worldbody><body name="root"><freejoint/><geom size=".1"/><body name="arm" pos="0 0 .3"><joint name="elbow"/><geom size=".1"/></body></body></worldbody></mujoco>')
    q=np.broadcast_to(model.qpos0,(4,model.nq)).copy();q[:,:3]=[[1,2,3],[2,3,4],[3,4,5],[4,5,6]];q[:,-1]=[.1,.2,.3,.4]
    path=tmp_path/'gpu_qpos.npz';np.savez(path,qpos=q,fps=30)
    loaded,names,values=native_motion_state(model,path,4)
    np.testing.assert_array_equal(loaded,q)
    assert names==['elbow'];np.testing.assert_array_equal(values[:,0],q[:,-1])
    with pytest.raises(ValueError,match='finite'):native_motion_state(model,path,3)
    q[:,3]=2;np.savez(path,qpos=q,fps=30)
    with pytest.raises(ValueError,match='quaternion'):native_motion_state(model,path,4)


def test_supplied_family_and_reconstructed_h_have_distinct_provenance(tmp_path):
    body=tmp_path/'body';body.mkdir()
    (body/'input_manifest.json').write_text(json.dumps(dict(motion_source=dict(family='smplx',evidence='fitted'),policy='supplied-gripper-body-v1')))
    (body/'prior-ready.json').write_text(json.dumps(dict(policy='supplied-gripper-body-v1',human_model='smplx',body_route='supplied_body')))
    supplied=source_human_metadata(body)
    assert supplied['human_model']=='smplx' and supplied['body_route']=='supplied_body'
    assert supplied['source_body_evidence']=='fitted' and 'bypassed' in supplied['body_prior']
    (body/'input_manifest.json').write_text(json.dumps(dict(policy='gripper-head-two-ee-smplh-v2')))
    (body/'prior-ready.json').write_text(json.dumps(dict(policy='gripper-head-two-ee-smplh-v2')))
    (body/'human_reference.json').write_text(json.dumps(dict(family='smplh',policy='egoallo-smplh-native-only-v2')))
    inferred=source_human_metadata(body)
    assert inferred['human_model']=='smplh' and inferred['body_route']=='three-point-egoallo'
    assert inferred['source_body_evidence']=='reconstructed'
