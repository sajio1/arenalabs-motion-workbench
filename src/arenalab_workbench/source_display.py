"""Source-world display metadata and camera similarity; never alter robot poses.

Matrices map column vectors from the numerical Z-up world to a display world.
Clouds/cameras already exported in that display basis only need the similarity.
Raw source body/object geometry additionally needs the declared basis rotation.
"""
import numpy as np

MANIPULATION_WORLD_TO_DISPLAY = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
GRIPPER_WORLD_TO_DISPLAY = [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]


def display_basis(value=None):
    matrix = np.asarray(MANIPULATION_WORLD_TO_DISPLAY if value is None else value, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-8, rtol=0)
            or not np.allclose(matrix[:, 2], [0., 1., 0.], atol=1e-8, rtol=0)):
        raise ValueError('Invalid Z-up to Y-up source display rotation')
    return matrix


def source_similarity(metadata=None, world_to_display=None):
    basis = display_basis(world_to_display)
    if not metadata or metadata.get('applied') is True:
        return 1., np.zeros(3)
    scale = float(metadata['scale'])
    anchor = np.asarray(metadata['anchor_xyz_m'], dtype=float)
    if (metadata.get('world_up') != 'z' or not np.isfinite(scale) or scale <= 0
            or anchor.shape != (3,) or not np.isfinite(anchor).all()):
        raise ValueError('Invalid source similarity transform')
    return scale, (1. - scale) * (basis @ anchor)


def source_camera_similarity(poses, metadata=None, world_to_display=None):
    """Return copied camera poses + depth multiplier; rotations stay identical.

The input camera is already in the renderer's display basis. Its intrinsics and
RGB remain unchanged. Scale each source depth frame by the returned multiplier,
without mutating the shared depth cache. Robot geometry is already transformed.
"""
    scale, offset = source_similarity(metadata, world_to_display)
    result = np.array(poses, dtype=float, copy=True)
    if result.shape[-2:] != (4, 4) or not np.isfinite(result).all():
        raise ValueError('Invalid source camera poses')
    result[..., :3, 3] = scale * result[..., :3, 3] + offset
    return result, scale


def source_scene_metadata(preview, numerical, world_to_display):
    result = {'source_world_to_display': display_basis(world_to_display).tolist()}
    if preview:
        result['source_body'] = preview
    if numerical.get('trajectory_scale') == 'match_robot':
        transform = numerical['scale_transform']
        source_similarity(transform, world_to_display)
        result['source_transform'] = dict(transform, applied=False)
    return result
