"""Read lossless cloud/segmentation extensions from an exported dataset.

Example: sample = cloud_at('/path/to/dataset', frame_index=100); xyz = sample['xyz']
The returned source_frame describes the actual observation, without interpolation.
"""
from pathlib import Path
import json
import numpy as np


def cloud_at(dataset, frame_index):
    root = Path(dataset)
    scene = json.loads((root/'extensions/arenalabs/scene.json').read_text())
    cloud = scene.get('cloud')
    if not cloud or not 0 <= frame_index < scene['frames']:
        raise ValueError('Point cloud unavailable or frame outside episode')
    frames = cloud['frames']
    index = int(np.searchsorted([f['source_frame'] for f in frames], frame_index, side='right')-1)
    if index < 0:
        raise ValueError('No earlier cloud observation for this frame')
    frame = frames[index]
    path = root/cloud['url']
    def field(name, dtype, size):
        return np.fromfile(path, dtype=dtype, count=frame['count']*size, offset=frame[name]).reshape(frame['count'],size)
    return dict(xyz=field('positions','<f4',3), rgb=field('colors','u1',3), instance_ids=field('labels','u1',1)[:,0], source_frame=frame['source_frame'], timestamp=frame['time'])
