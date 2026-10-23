import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from arenalab_workbench.source_display import (
    MANIPULATION_WORLD_TO_DISPLAY, GRIPPER_WORLD_TO_DISPLAY,
    display_basis, source_camera_similarity, source_scene_metadata,
)


@pytest.mark.parametrize('basis', [MANIPULATION_WORLD_TO_DISPLAY, GRIPPER_WORLD_TO_DISPLAY])
@pytest.mark.parametrize('scale', [.58, 1., 1.4])
def test_scaled_world_camera_retains_rgb_projection_and_depth_order(basis, scale):
    """Independent pinhole projection of an asymmetric moving camera + scene.

    This detects using S for gripper C, rotating twice, leaving camera origins
    unchanged, or failing to scale the observed depths in the FOV compositor.
    """
    basis = np.array(basis)
    anchor = np.array([3.2, -.7, .25])
    raw_camera = np.repeat(np.eye(4)[None], 2, axis=0)
    raw_camera[:, :3, :3] = Rotation.from_euler('xyz', [[.1,.3,-.5],[-.2,.4,.2]]).as_matrix()
    raw_camera[:, :3, 3] = [[1.,2.,.8],[1.1,1.9,.85]]
    camera = raw_camera.copy()
    camera[:, :3, :3] = basis @ raw_camera[:, :3, :3]
    camera[:, :3, 3] = raw_camera[:, :3, 3] @ basis.T
    original = camera.copy()
    camera_points = np.array([[.2,-.1,1.3],[-.3,.4,2.1]])
    points = np.einsum('tij,pj->tpi', raw_camera[:, :3,:3], camera_points) + raw_camera[:,None,:3,3]
    original_display = points @ basis.T
    scaled_display = (anchor + scale * (points-anchor)) @ basis.T
    transformed, depth_scale = source_camera_similarity(camera, dict(scale=scale, anchor_xyz_m=anchor.tolist(), world_up='z'), basis)
    before = np.einsum('tji,tpj->tpi', camera[:,:3,:3], original_display-camera[:,None,:3,3])
    after = np.einsum('tji,tpj->tpi', transformed[:,:3,:3], scaled_display-transformed[:,None,:3,3])
    K = np.array([[580.,0.,480.],[0.,560.,270.],[0.,0.,1.]])
    old_uv, new_uv = before @ K.T, after @ K.T
    np.testing.assert_allclose(old_uv[...,:2]/old_uv[...,2:], new_uv[...,:2]/new_uv[...,2:], atol=1e-10)
    np.testing.assert_allclose(after[...,2], depth_scale*before[...,2], atol=1e-12)
    np.testing.assert_array_equal(transformed[:,:3,:3], camera[:,:3,:3])
    np.testing.assert_array_equal(camera, original)
    # Source occlusion epsilon/validity are in the scaled scene's units.
    measured=np.array([0., .01, 1., 1., 1.])
    z=np.array([1.,1.,.9,1.01,1.02])
    old_keep=~((measured>.03)&(z>measured+.012))
    new_keep=~((measured*scale>.03*scale)&(z*scale>measured*scale+.012*scale))
    np.testing.assert_array_equal(old_keep,new_keep)


def test_metadata_exports_raw_source_body_basis_and_single_similarity():
    preview={'vertices_url':'/vertices.bin','world_up':'z','source_uniform_similarity_applied':False}
    numerical={'trajectory_scale':'match_robot','scale_transform':{'scale':.7,'anchor_xyz_m':[1,2,.1],'world_up':'z'}}
    result=source_scene_metadata(preview,numerical,GRIPPER_WORLD_TO_DISPLAY)
    assert result['source_body'] == preview
    assert result['source_world_to_display'] == GRIPPER_WORLD_TO_DISPLAY
    assert result['source_transform']['applied'] is False
    assert 'applied' not in numerical['scale_transform']
    identity,scale=source_camera_similarity(np.eye(4),dict(result['source_transform'],applied=True),GRIPPER_WORLD_TO_DISPLAY)
    np.testing.assert_array_equal(identity,np.eye(4))
    assert scale == 1.
    assert 'source_transform' not in source_scene_metadata(preview,{'trajectory_scale':'preserve'},GRIPPER_WORLD_TO_DISPLAY)


def test_display_basis_rejects_scale_reflection_or_wrong_up_axis():
    for value in [np.eye(3),np.eye(3)*2,np.diag([1,1,-1]),np.ones((2,3)),np.full((3,3),np.nan)]:
        with pytest.raises(ValueError):display_basis(value)
    with pytest.raises(ValueError):source_camera_similarity(np.eye(4),dict(scale=-1,anchor_xyz_m=[0,0,0],world_up='z'))
