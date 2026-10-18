"""Fixed semantic target point derived from the rigid gripper housing CAD."""
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

def housing_reference(directory, description):
    root = next(n for n in description['nodes'] if n['name'] == description['root'])
    vertices = []
    for visual in root['visuals']:
        mesh = trimesh.load(directory / visual['mesh'], force='mesh') if 'mesh' in visual else trimesh.creation.box(visual['box'])
        points = np.asarray(mesh.vertices) * visual.get('scale', [1, 1, 1])
        points = points @ Rotation.from_euler('xyz', visual['rpy']).as_matrix().T + visual['xyz']
        vertices.append(points)
    points = np.concatenate(vertices)
    bounds = np.stack([points.min(0), points.max(0)])
    center = bounds.mean(0)
    return center, dict(policy='rigid-housing-center-v1', link=description['root'], position_in_mount_m=center.tolist(), bounds_in_mount_m=bounds.tolist(), definition='Center of the fixed housing visual bounding box; excludes moving fingers', source_pose_edited=False)
