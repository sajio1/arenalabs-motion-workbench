"""Exact export replay conventions and rejection of ambiguous/corrupt inputs."""
import json
import numpy as np
import pytest
from arenalab_workbench.dataset_io.replay import build_replay, replay_arrays, Y_UP_TO_Z_UP


def motion():
    return dict(fps=30,frames=3,positions=[[[1,2,3]],[[1.1,2,3]],[[1.2,2,3]]],
                quaternions=[[[0,0,0,1]],[[0,0,0,-1]],[[0,0,0,1]]])


def test_basis_and_quaternion_hemisphere():
    assert np.allclose(Y_UP_TO_Z_UP@[1,2,3],[1,-3,2])
    assert np.linalg.det(Y_UP_TO_Z_UP)==1
    positions, quaternions=replay_arrays(motion())
    assert np.allclose(quaternions[:,:,3],1)
    assert np.allclose(positions[:,0,0],[1,1.1,1.2])


def test_replay_rejects_invalid_quaternion_and_clock():
    data=motion();data['quaternions'][0][0]=[0,0,0,0]
    with pytest.raises(ValueError,match='unit length'):replay_arrays(data)
    data=motion();data['fps']=60
    with pytest.raises(ValueError,match='30 Hz'):replay_arrays(data)


def test_replay_contains_geometry_motion_and_no_external_paths(tmp_path):
    trimesh=pytest.importorskip('trimesh')
    scene=trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(),node_name='geom_0',geom_name='geom_0')
    model=tmp_path/'robot.glb';model.write_bytes(scene.export(file_type='glb'))
    result=build_replay(model,motion(),tmp_path/'replay')
    text=(tmp_path/'replay/robot.usda').read_text(encoding='utf-8')
    assert 'endTimeCode = 2' in text and 'timeCodesPerSecond = 30' in text
    assert '0: (1, 2, 3)' in text and '2: (1.2, 2, 3)' in text
    assert '0: (1, 0, 0, 0)' in text  # USD wxyz, not GLB xyzw.
    assert 'faceVertexIndices' in text and '@' not in text
    assert not result['physics_articulation']
    assert json.loads((tmp_path/'replay/replay.json').read_text())['meshes']==1
    assert (tmp_path/'replay/isaac_replay.py').is_file()


def test_replay_rejects_mesh_identity_mismatch(tmp_path):
    trimesh=pytest.importorskip('trimesh')
    scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),node_name='geom_5')
    model=tmp_path/'robot.glb';model.write_bytes(scene.export(file_type='glb'))
    with pytest.raises(ValueError,match='identities differ'):
        build_replay(model,motion(),tmp_path/'replay')
