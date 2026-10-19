"""Read lossless cloud/segmentation extensions from an exported dataset.

Example: sample = cloud_at('/path/to/dataset', frame_index=100); xyz = sample['xyz']
The returned source_frame describes the actual observation, without interpolation.
"""
from pathlib import Path
import json
from functools import lru_cache
from collections import OrderedDict
from threading import Lock
import numpy as np


def decode_rle(value):
    """COCO-style uncompressed runs, column-major, beginning with a zero run."""
    runs=np.asarray(value['counts'],dtype='i8')
    if (runs<0).any() or runs.sum()!=int(np.prod(value['size'])):
        raise ValueError('Malformed segmentation RLE')
    return np.repeat(np.arange(len(runs))%2,runs).reshape(value['size'],order='F').astype(bool)


@lru_cache(maxsize=8)
def _geometry_index(folder):
    return json.loads((Path(folder)/'index.json').read_text())


@lru_cache(maxsize=32)
def _camera_group(folder, group):
    import pyarrow.parquet as pq
    return pq.ParquetFile(Path(folder)/'cameras.parquet').read_row_group(group).to_pylist()


@lru_cache(maxsize=8)
def _depth_array(folder):
    import zarr
    return zarr.open_group(str(Path(folder)/'depth.zarr'),mode='r')['depth_mm']


class _ProxyFrames:
    """Bounded sequential decode; opening a codec for each frame is too expensive."""
    def __init__(self,folder):
        import cv2
        self.cv2=cv2;self.lock=Lock();self.frames=OrderedDict();self.position=0
        self.length=len(_geometry_index(folder)['frames'])
        path=str(Path(folder)/'rgb.mp4')
        if hasattr(cv2,'CAP_PROP_N_THREADS'):
            self.cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        else:self.cap=cv2.VideoCapture(path)
        if not self.cap.isOpened():raise ValueError('Geometry RGB proxy unavailable')

    def get(self,index):
        with self.lock:
            if index in self.frames:
                self.frames.move_to_end(index);return self.frames[index]
            if self.position!=index:
                self.cap.set(self.cv2.CAP_PROP_POS_FRAMES,index);self.position=index
            for i in range(index,min(index+8,self.length)):
                ok,image=self.cap.read()
                if not ok:raise ValueError('Geometry RGB proxy coverage mismatch')
                self.position=i+1;self.frames[i]=self.cv2.cvtColor(image,self.cv2.COLOR_BGR2RGB)
            while len(self.frames)>24:self.frames.popitem(last=False)
            return self.frames[index]

    def __del__(self):
        if hasattr(self,'cap'):self.cap.release()


_proxy_readers=OrderedDict()
_proxy_lock=Lock()


def _proxy_frames(folder):
    with _proxy_lock:
        if folder not in _proxy_readers:_proxy_readers[folder]=_ProxyFrames(folder)
        _proxy_readers.move_to_end(folder)
        while len(_proxy_readers)>4:_proxy_readers.popitem(last=False)
        return _proxy_readers[folder]


def geometry_frame(folder, index):
    """Read one canonical RGB-D observation; no reconstruction model required."""
    import cv2
    folder=Path(folder)
    info=_geometry_index(str(folder))
    if not 0<=index<len(info['frames']):raise ValueError('Geometry frame outside recording')
    row=info['frames'][index]
    depth=np.asarray(_depth_array(str(folder))[index])
    group_size=info.get('camera_row_group_size',256)
    cameras=_camera_group(str(folder),index//group_size)[index%group_size]
    with (folder/'masks.rle.jsonl').open('rb') as stream:
        stream.seek(row['mask_offset']);masks=json.loads(stream.read(row['mask_bytes']))
    labels=np.zeros(depth.shape,dtype='u2')
    for item in masks['instances']:labels[decode_rle(item['rle'])]=item['label']
    image=_proxy_frames(str(folder)).get(index)
    if image.shape[:2]!=depth.shape:raise ValueError('Geometry RGB/depth pixel-space mismatch')
    return dict(depth_mm=depth,rgb=image,labels=labels,
        intrinsic=np.asarray(cameras['intrinsic']).reshape(3,3),
        camera_to_world=np.asarray(cameras['camera_to_world']).reshape(4,4),
        source_frame=int(cameras['source_frame']),timestamp=float(cameras['timestamp_s']),
        display_from_source=np.asarray(info['display_from_source']))


def points_from_geometry(value, maximum_points=None):
    depth=value['depth_mm'].astype('f4')*.001
    yy,xx=np.indices(depth.shape);valid=depth>0
    selected=np.flatnonzero(valid)
    if maximum_points is not None and len(selected)>maximum_points:
        selected=selected[np.linspace(0,len(selected)-1,maximum_points,dtype=int)]
    rays=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3)[selected]@np.linalg.inv(value['intrinsic']).T
    transform=value['display_from_source']@value['camera_to_world']
    xyz=(rays*depth.ravel()[selected,None])@transform[:3,:3].T+transform[:3,3]
    return dict(xyz=xyz.astype('f4'),rgb=value['rgb'].reshape(-1,3)[selected],
        instance_ids=value['labels'].ravel()[selected],source_frame=value['source_frame'],timestamp=value['timestamp'])


def cloud_at(dataset, frame_index, maximum_points=None):
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
    if cloud.get('format')=='depth-rgb-v1':
        return points_from_geometry(geometry_frame(root/cloud['geometry_url'],index),maximum_points)
    path = root/cloud['url']
    def field(name, dtype, size):
        return np.fromfile(path, dtype=dtype, count=frame['count']*size, offset=frame[name]).reshape(frame['count'],size)
    return dict(xyz=field('positions','<f4',3), rgb=field('colors','u1',3), instance_ids=field('labels','u1',1)[:,0], source_frame=frame['source_frame'], timestamp=frame['time'])
