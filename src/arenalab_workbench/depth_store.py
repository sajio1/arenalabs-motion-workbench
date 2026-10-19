"""Lazy, frame-addressable depth contract shared by perception and compositors.

Raw model depth is float32 metres. Canonical delivery depth is uint16 mm;
it never replaces the immutable raw prediction. Legacy NPZ remains readable.
"""
from pathlib import Path
import json
import hashlib
from functools import lru_cache
import numpy as np

MODEL = 'facebook/map-anything-apache'
POLICY = 'mapanything-camera-depth-v1'


def fov_current(scene, robot):
    if scene.get('patch_unavailable',{}).get(robot):return True
    if not scene.get('patch_videos',{}).get(robot):return False
    if scene.get('patch_kinds',{}).get(robot)=='native_preview':return True
    policy=scene.get('geometry',{}).get('depth',{}).get('policy')
    return policy!=POLICY or scene.get('patch_depth_policy',{}).get(robot)==policy


def exists(work):
    work = Path(work)
    return (work/'depth-store.json').is_file() or (work/'depth.npz').is_file()


def current(work):
    work=Path(work);path=work/'depth.json'
    if not (exists(work) and path.is_file() and json.loads(path.read_text()).get('policy')==POLICY):return False
    marker=work/'depth-store.json'
    if not marker.exists():return False
    run=(work/json.loads(marker.read_text())['path']).parent
    receipt=run/'input.json'
    if not receipt.exists():return False
    expected=json.loads(receipt.read_text())['input_sha256']
    names={'frames.npz','cameras.npz','camera-poses.npz','camera-intrinsics.npz'}
    actual={name:hashlib.sha256((work/name).read_bytes()).hexdigest() for name in names if (work/name).is_file()}
    return actual=={name:value for name,value in expected.items() if name in names}


def create_array(group, name, shape, dtype, chunks=None):
    from numcodecs import Blosc
    return group.create_dataset(name, shape=shape, chunks=chunks or (1, *shape[1:]),
        dtype=dtype, compressor=Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE))


class FrameImages:
    def __init__(self, work, arrays):
        self.work, self.arrays = Path(work), arrays
        self.names = sorted((self.work/'frames').glob('*.jpg'))
        self.shape = (*arrays['depth'].shape, 3)

    def __getitem__(self, index):
        import cv2
        if not isinstance(index, (int, np.integer)):
            return np.stack([self[int(i)] for i in np.arange(self.shape[0])[index]])
        image = cv2.imread(str(self.names[index]))
        if image is None:
            raise ValueError('Prepared RGB observation is missing')
        h, w = self.shape[1:3]
        affine = self.arrays['preprocessing_affine'][index]
        return cv2.cvtColor(cv2.warpAffine(image, affine[:2], (w, h),
            flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)


class NativeWindowArray:
    """Read original selected predictions without duplicating them in another store."""
    def __init__(self, folder, metadata, field):
        self.folder,self.field=Path(folder),field
        self.shape=tuple(metadata['native_shape']);self.dtype=np.dtype('f4')
        self.owners=metadata['native_owners'];self.locals=metadata['native_locals']
        self.files=metadata['native_chunks']
        self._cached=lru_cache(maxsize=2)(self._read)

    def _read(self,chunk):
        relative=Path(self.files[chunk])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Native depth path escapes run')
        path=self.folder/relative
        if path.is_dir():
            import zarr
            value=zarr.open_group(str(path),mode='r')[self.field][:]
        else:
            with np.load(path) as saved:value=saved[self.field]
        value.setflags(write=False)
        return value

    def __getitem__(self,index):
        if isinstance(index,(int,np.integer)):
            index=int(index)
            if index<0:index+=self.shape[0]
            if not 0<=index<self.shape[0]:raise IndexError('Depth frame outside sequence')
            return self._cached(self.owners[index])[self.locals[index]]
        return np.stack([self[int(i)] for i in np.arange(self.shape[0])[index]])

    def __len__(self):return self.shape[0]


def load(work):
    """Arrays support frame indexing; callers must not inflate the whole sequence."""
    work = Path(work)
    marker = work/'depth-store.json'
    if marker.exists():
        import zarr
        metadata = json.loads(marker.read_text())
        relative = Path(metadata['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Depth store escapes its workspace')
        # Trusted worker snapshots/compositors may symlink a shared immutable run.
        location = work/relative
        group = zarr.open_group(str(location), mode='r')
        arrays = {key: group[key] for key in group.array_keys()}
        if 'depth' not in arrays:
            metadata=dict(group.attrs)
            arrays['depth']=NativeWindowArray(location.parent,metadata,'depth')
            if metadata.get('native_confidence'):arrays['conf']=NativeWindowArray(location.parent,metadata,'conf')
        arrays['images'] = FrameImages(work, arrays)
        return arrays
    with np.load(work/'depth.npz') as saved:
        # Legacy archives require one decode, never one decode per video frame.
        return {key: saved[key] for key in saved.files}


def millimetres(depth):
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth >= .0005) & (depth <= 65.535)
    result = np.zeros(depth.shape, dtype='<u2')
    result[valid] = np.rint(depth[valid]*1000).astype('<u2')
    return result
