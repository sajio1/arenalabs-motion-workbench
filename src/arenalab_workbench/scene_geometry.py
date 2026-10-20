"""Canonical current-frame RGB-D storage and compact frame delivery.

The permanent geometry bundle has no XYZ buffers, object meshes or fused map.
Raw model predictions remain private, separate and immutable.
"""
from pathlib import Path
import json
import struct
from functools import lru_cache
from collections import OrderedDict
from threading import Lock
import numpy as np
from arenalab_workbench.depth_store import create_array, millimetres
from arenalab_workbench.dataset_io.portable_reader import geometry_frame, points_from_geometry


@lru_cache(maxsize=64)
def _cloud_reference(path,modified_ns,size):
    return json.loads(Path(path).read_text())['cloud']['geometry_url']


def encode_rle(mask):
    flat=np.asarray(mask,dtype='u1').ravel(order='F')
    edges=np.flatnonzero(np.diff(np.r_[0,flat,0]))
    counts=np.diff(np.r_[0,edges,len(flat)]).tolist()
    # A terminating zero run is legal and makes the encoding deterministic.
    return dict(size=list(mask.shape),counts=counts)


class Writer:
    def __init__(self, folder, shape, display_from_source, source_human=False):
        import zarr
        import imageio.v2 as imageio
        self.folder=Path(folder)
        if self.folder.exists():raise FileExistsError('Geometry bundles are immutable; choose a fresh directory')
        self.folder.mkdir(parents=True)
        t,h,w=shape
        self.depth=create_array(zarr.open_group(str(self.folder/'depth.zarr'),mode='w'),
            'depth_mm',shape,'<u2')
        self.depth.attrs.update(units='millimetres',invalid_value=0,maximum_rounding_error_m=.0005)
        self.human_depth=create_array(zarr.open_group(str(self.folder/'depth.zarr'),mode='a'),'human_depth_mm',shape,'<u2') if source_human else None
        self.video=imageio.get_writer(self.folder/'rgb.mp4',fps=15,codec='libx264',
            quality=9,macro_block_size=1,ffmpeg_params=['-threads','2','-pix_fmt','yuv420p'])
        self.masks=(self.folder/'masks.rle.jsonl').open('wb')
        self.rows=[];self.frames=[];self.expected=t;self.display=np.asarray(display_from_source)

    def append(self,index,depth,rgb,intrinsic,extrinsic,semantic,valid,source_frame,affine=None,human_valid=None):
        if index!=len(self.frames):raise ValueError('Geometry frames must follow the source clock')
        mm=millimetres(depth);mm[~valid]=0
        self.depth[index]=mm
        if self.human_depth is not None:
            human_mm=millimetres(depth);human_mm[~np.asarray(human_valid,dtype=bool)]=0
            self.human_depth[index]=human_mm
        self.video.append_data(np.asarray(rgb,dtype='u1'))
        payload=dict(source_frame=int(source_frame),instances=[dict(label=int(label),rle=encode_rle(semantic==label))
            for label in np.unique(semantic) if label])
        encoded=(json.dumps(payload,separators=(',',':'))+'\n').encode()
        self.frames.append(dict(source_frame=int(source_frame),time=float(source_frame/30),
            count=int(np.count_nonzero(mm)),human_count=int(np.count_nonzero(human_mm)) if self.human_depth is not None else 0,positions=0,colors=0,labels=0,
            mask_offset=self.masks.tell(),mask_bytes=len(encoded)))
        self.masks.write(encoded)
        e=np.eye(4);e[:3]=extrinsic
        self.rows.append(dict(source_frame=int(source_frame),timestamp_s=float(source_frame/30),
            intrinsic=np.asarray(intrinsic).ravel().tolist(),camera_to_world=np.linalg.inv(e).ravel().tolist(),
            preprocessing_affine=None if affine is None else np.asarray(affine).ravel().tolist()))

    def finish(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.video.close();self.masks.close()
        if len(self.frames)!=self.expected:raise ValueError('Incomplete geometry bundle')
        pq.write_table(pa.Table.from_pylist(self.rows),self.folder/'cameras.parquet',compression='zstd',row_group_size=256)
        (self.folder/'index.json').write_text(json.dumps(dict(schema='rgbd-geometry-v1',fps=15,
            depth='depth.zarr/depth_mm',depth_units='millimetres',invalid_value=0,
            rgb='rgb.mp4',rgb_semantics='Encoded proxy of aligned source RGB; not novel-view synthesis',
            cameras='cameras.parquet',camera_row_group_size=256,masks='masks.rle.jsonl',frames=self.frames,
            display_from_source=self.display.tolist(),point_cloud_stored=False,
            human_and_hardware_excluded=True,history_fused=False,
            source_human_layer='depth.zarr/human_depth_mm' if self.human_depth is not None else None),separators=(',',':')))
        return self.frames


def frame_packet(folder,index,rgb_encoding='raw'):
    value=geometry_frame(folder,index)
    if rgb_encoding=='jpeg':
        import cv2
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(value['rgb'],cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])
        if not ok:raise ValueError('RGB preview encoding failed')
        rgb=encoded.tobytes()
    elif rgb_encoding=='raw':rgb=value['rgb'].astype('u1').tobytes()
    else:raise ValueError('Unknown RGB preview encoding')
    label_bytes=1 if value['labels'].max(initial=0)<256 else 2
    labels=value['labels'].astype('u1' if label_bytes==1 else '<u2').tobytes()
    human=value.get('human_depth_mm')
    human_bytes=0 if human is None else human.size*2
    transform=value['display_from_source']@value['camera_to_world']
    header=json.dumps(dict(shape=list(value['depth_mm'].shape),intrinsic=value['intrinsic'].tolist(),
        camera_to_world=transform.tolist(),depth_to_display=transform.tolist(),transform_semantics='Registered depth-to-display affine; physical optical poses remain in camera Parquet',source_frame=value['source_frame'],timestamp=value['timestamp'],
        depth_scale=.001,rgb_encoding=rgb_encoding,rgb_bytes=len(rgb),label_bytes=label_bytes,human_depth_bytes=human_bytes),separators=(',',':')).encode()
    header+=b' '*((-len(header))%4)
    return b''.join([struct.pack('<I',len(header)),header,value['depth_mm'].astype('<u2').tobytes(),
        rgb,labels,b'' if human is None else human.astype('<u2').tobytes()])


class PreviewPackets:
    """Disposable wire-response cache; canonical depth and masks stay untouched."""
    def __init__(self,maximum_bytes=128*1024*1024):
        self.maximum_bytes=maximum_bytes;self.bytes=0;self.values=OrderedDict();self.lock=Lock()

    def get(self,folder,index,rgb,compressed):
        import gzip
        key=(str(folder),index,rgb,compressed)
        with self.lock:
            if key in self.values:
                self.values.move_to_end(key);return self.values[key]
        value=frame_packet(folder,index,rgb)
        if compressed:value=gzip.compress(value,compresslevel=1)
        with self.lock:
            if key in self.values:return self.values[key]
            if len(value)<=self.maximum_bytes:
                self.values[key]=value;self.bytes+=len(value)
                while self.bytes>self.maximum_bytes:
                    _,expired=self.values.popitem(last=False);self.bytes-=len(expired)
        return value


_preview_packets=PreviewPackets()


def attach_routes(app):
    from fastapi import Request,HTTPException,Response
    from arenalab_workbench.dataset_io import storage
    from urllib.parse import urlsplit

    @app.get('/geometry/frame')
    def get_frame(scene: str, frame: int, request: Request,rgb: str='raw',revision: str=''):
        try:
            parsed=urlsplit(scene)
            if parsed.scheme or parsed.netloc or not parsed.path.startswith('/assets/'):raise ValueError('Local scene required')
            path=storage.within(storage.STORE,parsed.path.removeprefix('/assets/'))
            if path.name!='scene.json' or not storage.permitted(path,request.cookies.get(storage.COOKIE)):
                raise ValueError('Scene unavailable')
            stat=path.stat();url=_cloud_reference(str(path),stat.st_mtime_ns,stat.st_size)
            if not url.startswith('/assets/'):raise ValueError('Local geometry required')
            if revision and revision!=url:raise ValueError('Geometry revision changed; reload scene')
            geometry=storage.within(storage.STORE,url.removeprefix('/assets/'))
            if not storage.permitted(geometry,request.cookies.get(storage.COOKIE)):
                raise ValueError('Geometry unavailable')
            compressed='gzip' in request.headers.get('accept-encoding','')
            packet=_preview_packets.get(geometry,frame,rgb,compressed)
        except (ValueError,KeyError,FileNotFoundError,IndexError) as error:
            raise HTTPException(404,'Geometry observation unavailable') from error
        # Revision is the immutable geometry URL; a new alignment gets a new key.
        headers={'Cache-Control':'private, max-age=300' if revision else 'private, no-store','Vary':'Accept-Encoding, Cookie'}
        if compressed:headers['Content-Encoding']='gzip'
        return Response(packet,media_type='application/octet-stream',headers=headers)
