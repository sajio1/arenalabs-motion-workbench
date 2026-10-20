"""Geometry delivery must preserve calibration, source clocks and workspace privacy."""
import json
import struct
import numpy as np
import pytest
from arenalab_workbench.depth_store import millimetres
from arenalab_workbench.scene_geometry import encode_rle
from arenalab_workbench.dataset_io.portable_reader import decode_rle,geometry_frame,points_from_geometry
from arenalab_workbench.mapanything_depth import windows,selection


def test_old_fov_is_not_reused_for_new_geometry():
    from arenalab_workbench.depth_store import POLICY,fov_current
    scene=dict(geometry=dict(depth=dict(policy=POLICY)),patch_videos=dict(h2='/old.mp4'))
    assert not fov_current(scene,'h2')
    scene['patch_depth_policy']={'h2':POLICY}
    assert fov_current(scene,'h2')
    scene['patch_videos']={};scene['patch_unavailable']={'h2':'Uncalibrated optical camera'}
    assert fov_current(scene,'h2')


def test_depth_range_and_quantization():
    source=np.array([0,np.nan,np.inf,-1,.0001,.12349,1.23451,65.535,65.536])
    before=source.copy()
    encoded=millimetres(source)
    assert encoded.tolist()==[0,0,0,0,0,123,1235,65535,0]
    np.testing.assert_equal(source,before)
    assert np.max(abs(encoded[5:8]*.001-source[5:8]))<=.0005


@pytest.mark.parametrize('mask',[np.zeros((5,7)),np.ones((5,7)),np.eye(6),np.arange(35).reshape(5,7)%3==0])
def test_mask_rle_exact(mask):
    np.testing.assert_array_equal(decode_rle(encode_rle(mask)),mask.astype(bool))


def test_windows_cover_every_source_frame_without_averaging():
    for n in [1,17,32,33,47,133,301,1301]:
        chunks=windows(n);owners=selection(chunks,n)
        assert len(owners)==n
        assert all(i in chunks[j] for i,j in enumerate(owners))
        assert all(len(c)<=32 for c in chunks)


@pytest.mark.parametrize('format',['npz','zarr'])
def test_native_windows_preserve_selected_original_predictions(tmp_path,format):
    from arenalab_workbench.depth_store import NativeWindowArray,create_array
    if format=='zarr':zarr=pytest.importorskip('zarr')
    for j in range(2):
        value=np.broadcast_to((100+j*100+np.arange(4))[:,None,None],(4,2,3)).astype('f4')
        if format=='npz':np.savez_compressed(tmp_path/f'{j}.npz',depth=value)
        else:create_array(zarr.open_group(str(tmp_path/f'{j}.zarr'),mode='w'),'depth',value.shape,value.dtype,chunks=value.shape)[:]=value
    meta=dict(native_shape=[6,2,3],native_owners=[0,0,0,1,1,1],native_locals=[0,1,2,1,2,3],native_chunks=[f'{j}.{format}' for j in range(2)])
    array=NativeWindowArray(tmp_path,meta,'depth')
    assert [float(array[i][0,0]) for i in range(6)]==[100,101,102,201,202,203]
    assert float(array[-1][0,0])==203
    with pytest.raises(ValueError):array[0][0,0]=9
    assert array._cached.cache_info().currsize<=2


@pytest.fixture
def geometry(tmp_path):
    pytest.importorskip('zarr');pytest.importorskip('pyarrow');pytest.importorskip('imageio')
    from arenalab_workbench.scene_geometry import Writer
    folder=tmp_path/'geometry';shape=(2,16,16)
    k=np.array([[100.,2,8],[0,120.,7],[0,0,1]])
    c2w=np.eye(4);c2w[:3,3]=[.3,-.2,1.4]
    rgb=np.full((16,16,3),120,dtype='u1');sem=np.zeros((16,16),'u1');sem[4:8,4:8]=3
    valid=np.ones((16,16),bool);valid[:2]=False
    writer=Writer(folder,shape,np.eye(4))
    for i in range(2):writer.append(i,np.full((16,16),1.2344),rgb,k,np.linalg.inv(c2w)[:3],sem,valid,i*2,np.eye(3))
    writer.finish()
    return folder,k,c2w,sem,valid


def test_rgbd_roundtrip_and_reprojection(geometry):
    folder,k,c2w,sem,valid=geometry
    value=geometry_frame(folder,1);points=points_from_geometry(value)
    assert value['source_frame']==2 and value['timestamp']==2/30
    np.testing.assert_array_equal(value['labels'],sem)
    np.testing.assert_array_equal(value['intrinsic'],k)
    np.testing.assert_allclose(value['camera_to_world'],c2w,atol=1e-14)
    assert len(points['xyz'])==valid.sum()
    local=(points['xyz']-c2w[:3,3])@c2w[:3,:3]
    projected=local@k.T;uv=projected[:,:2]/projected[:,2:]
    yy,xx=np.indices(valid.shape)
    np.testing.assert_allclose(uv,np.stack([xx[valid],yy[valid]],-1),atol=1e-4)
    assert not list(folder.rglob('*.bin'))
    assert not list(folder.rglob('*.ply'))


def test_frame_api_and_private_ownership(geometry,tmp_path,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from arenalab_workbench.scene_geometry import attach_routes
    from arenalab_workbench.dataset_io import storage
    folder,*_=geometry
    monkeypatch.setattr(storage,'STORE',tmp_path)
    scene=tmp_path/'scene.json';scene.write_text(json.dumps(dict(cloud=dict(geometry_url='/assets/geometry'))))
    app=FastAPI();attach_routes(app);client=TestClient(app)
    response=client.get('/geometry/frame',params=dict(scene='/assets/scene.json',frame=0))
    assert response.status_code==200,response.text
    n=struct.unpack('<I',response.content[:4])[0]
    header=json.loads(response.content[4:4+n]);assert header['source_frame']==0
    assert len(response.content)==4+n+16*16*6
    storage.mark_private(tmp_path,storage.owner('owner-token'))
    assert client.get('/geometry/frame',params=dict(scene='/assets/scene.json',frame=0)).status_code==404
    client.cookies.set(storage.COOKIE,'owner-token')
    assert client.get('/geometry/frame',params=dict(scene='/assets/scene.json',frame=0)).status_code==200
    assert client.get('/geometry/frame',params=dict(scene='/assets/scene.json',frame=2)).status_code==404
    assert client.get('/geometry/frame',params=dict(scene='/assets/../secret/scene.json',frame=0)).status_code==404


def test_source_human_cloud_is_separate_and_shares_whole_scene_transform(tmp_path):
    pytest.importorskip('zarr');pytest.importorskip('pyarrow');pytest.importorskip('imageio')
    from arenalab_workbench.scene_geometry import Writer,frame_packet
    folder=tmp_path/'with-human';shape=(1,16,16)
    transform=np.eye(4);transform[:3,:3]*=.8;transform[:3,3]=[.1,.2,-.1]
    k=np.array([[100.,0,8],[0,100.,8],[0,0,1]])
    human=np.zeros((16,16),bool);human[4:8,4:8]=True
    writer=Writer(folder,shape,transform,source_human=True)
    writer.append(0,np.ones((16,16)),np.full((16,16,3),100,dtype='u1'),k,np.eye(4)[:3],np.zeros((16,16),'u2'),~human,0,np.eye(3),human_valid=human)
    writer.finish()
    value=geometry_frame(folder,0)
    scene=points_from_geometry(value);hand=points_from_geometry(value,source_human=True)
    assert len(scene['xyz'])==240 and len(hand['xyz'])==16
    assert not scene['instance_ids'].any() and not hand['instance_ids'].any()
    np.testing.assert_allclose(hand['xyz'][:,2],.7,atol=1e-6)
    assert not (value['depth_mm']&value['human_depth_mm']).any()
    packet=frame_packet(folder,0);n=struct.unpack('<I',packet[:4])[0];header=json.loads(packet[4:4+n])
    assert header['human_depth_bytes']==16*16*2
    assert len(packet)==4+n+16*16*8
