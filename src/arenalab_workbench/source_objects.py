"""Validate supplied object geometry against its source motion's exact clock.

This checks the input container, not contact quality or physical feasibility.
It never fits, rescales, crops, interpolates or repairs supplied data.
"""
import numpy as np


def load_source_objects(path, timestamps_s):
    """Return validated arrays; body and object timestamps must be identical."""
    from .motion_sources import _npz, _finite, _rotation

    objects = _npz(path)
    required = {'object_ids', 'timestamps_s', 'positions_world_m', 'rotations_world'}
    missing = required - objects.keys()
    if missing:
        raise ValueError('Source object package is missing: ' + ', '.join(sorted(missing)))
    times = _finite(objects['timestamps_s'], 'object timestamps_s')
    if times.ndim != 1 or not np.array_equal(times, timestamps_s):
        raise ValueError('Object trajectories must use the exact source motion clock')
    names = objects['object_ids']
    if (names.ndim != 1 or names.dtype.kind not in 'US'
            or any(not str(name).strip() for name in names)
            or len(set(names.tolist())) != len(names)):
        raise ValueError('Source object_ids must be unique nonempty strings')
    count = len(names)
    positions = _finite(objects['positions_world_m'], 'object positions')
    rotations = _rotation(objects['rotations_world'], 'object rotations')
    if positions.shape != (len(times), count, 3) or rotations.shape != (len(times), count, 3, 3):
        raise ValueError('Object trajectories have invalid synchronized dimensions')
    for index in range(count):
        keys = {f'object_{index}_vertices_local_m', f'object_{index}_faces'}
        if keys - objects.keys():
            raise ValueError('Source object package is missing mesh arrays for object ' + str(index))
        vertices = _finite(objects[f'object_{index}_vertices_local_m'], 'object vertices')
        faces = objects[f'object_{index}_faces']
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Invalid embedded object mesh')
    return objects
