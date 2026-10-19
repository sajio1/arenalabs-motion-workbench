"""Numerical and storage invariants of the opt-in scene-asset path."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from arenalab_workbench.scene_assets.formats import (
    rle_encode, rle_decode, depth_mm, sparse_cloud, write_arrays, read_arrays, storage_budget,
)
from arenalab_workbench.scene_assets.geometry import (
    windows, robust_similarity, apply_transform, pose_graph, backproject,
)


@pytest.mark.parametrize("shape",[(3,7),(1,1),(8,4)])
def test_coco_rle_zero_first_column_major(shape):
    for value in [np.zeros(shape,bool),np.ones(shape,bool),np.indices(shape).sum(0)%2>0]:
        encoded=rle_encode(value)
        assert np.array_equal(rle_decode(encoded),value)
    assert rle_encode(np.array([[0,1],[0,1]]))["counts"]==[2,2]


def test_uint16_depth_never_overflows_or_converts_nan():
    values=np.array([0.,np.nan,np.inf,-1.,.0001,.1234,65.535,65.536])
    assert depth_mm(values).tolist()==[0,0,0,0,0,123,65535,0]


def test_sparse_points_reproject_and_have_no_duplicate_padding():
    depth=np.full((30,40),2.)
    k=np.array([[40,0,20],[0,40,15],[0,0,1.]])
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    sampled=sparse_cloud(depth,k,pose,4096,labels=np.full(depth.shape,300,dtype="uint16"))
    assert (sampled["instance_ids"]==300).all()
    cloud=sampled["xyz"]
    assert len(cloud)==1200
    assert len(np.unique(cloud,axis=0))==1200
    camera=cloud-pose[:3,3]
    uv=camera@k.T;uv=uv[:,:2]/uv[:,2:]
    np.testing.assert_allclose(uv,np.rint(uv),atol=1e-5)
    assert storage_budget(4096)==pytest.approx(7.962624)
    assert storage_budget(4096,fps=15)==pytest.approx(3.981312)


def test_windowing_covers_final_frame_and_overlap():
    chunks=windows(81,32,8)
    assert np.array_equal(np.unique(np.concatenate(chunks)),np.arange(81))
    assert all(len(np.intersect1d(a,b))==8 for a,b in zip(chunks,chunks[1:]))


def test_calibrated_depth_anchors_camera_without_registering_task_targets():
    pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec([.2,.1,.3]).as_matrix();pose[:3,3]=[1.,2.,3.]
    original=pose.copy()
    k=np.array([[70.,0,4],[0,70.,3],[0,0,1.]])
    depth=np.full((6,8),2.)
    world=backproject(depth,k,pose)
    camera=(world-pose[:3,3])@pose[:3,:3]
    np.testing.assert_allclose(camera[...,2],depth,atol=1e-12)
    np.testing.assert_array_equal(pose,original)


@pytest.mark.parametrize("metric",[True,False])
def test_static_registration_resists_dynamic_outliers(metric):
    rng=np.random.default_rng(4)
    points=rng.normal(size=(500,3))
    transform=np.eye(4)
    transform[:3,:3]=Rotation.from_rotvec([.1,.2,-.1]).as_matrix()*(1 if metric else 1.4)
    transform[:3,3]=[.2,-.4,.1]
    target=apply_transform(transform,points)
    target[:70]+=rng.normal(size=(70,3))*10
    fitted,audit=robust_similarity(points,target,metric)
    np.testing.assert_allclose(fitted,transform,atol=1e-8)
    assert audit["inliers"]==430


def test_pose_graph_loop_closure_reduces_drift():
    rng=np.random.default_rng(3)
    points=rng.normal(size=(100,3))
    b=np.eye(4);b[:3,3]=[1,0,0]
    c=np.eye(4);c[:3,3]=[2,0,0]
    drift=b.copy();drift[1,3]=.2
    end=c.copy();end[1,3]=.4
    edges=[(0,1,points,points-b[:3,3]),(1,2,points-b[:3,3],points-c[:3,3]),
           (0,2,points,points-c[:3,3])]
    solved,audit=pose_graph([np.eye(4),drift,end],edges)
    np.testing.assert_allclose(solved[2],c,atol=1e-7)
    assert audit["success"]
    with pytest.raises(ValueError,match="Disconnected"):
        pose_graph([np.eye(4),b,c],edges[:1])


def test_camera_parameter_parquet_preserves_poses_and_clock(tmp_path):
    pytest.importorskip("pyarrow")
    pose=np.tile(np.eye(4),(4,1,1));pose[:,0,3]=np.arange(4)/10
    path=tmp_path/"camera.parquet"
    write_arrays(path,{"pose":pose},np.array([0,2,4,6]))
    np.testing.assert_array_equal(read_arrays(path)["pose"],pose)
    import pyarrow.parquet as pq
    np.testing.assert_allclose(pq.read_table(path)["timestamp_s"].to_pylist(),np.arange(4)/15)


def test_expiry_never_removes_canonical_assets_or_unmarked_directories(tmp_path):
    from datetime import datetime,timezone
    from arenalab_workbench.scene_assets.formats import write_json
    from arenalab_workbench.scene_assets.retention import expire
    for name in ["assets","delivery","unowned"]:
        (tmp_path/name).mkdir()
        write_json(tmp_path/name/"delivery.json",dict(owned_temporary_delivery=name!="unowned",
                    expires_at="2026-01-01T00:00:00+00:00"))
    write_json(tmp_path/"assets/manifest.json",dict(robot_independent=True))
    candidates=expire(tmp_path,datetime(2026,10,19,tzinfo=timezone.utc))
    assert candidates==[str((tmp_path/"delivery").resolve())]
    assert (tmp_path/"delivery").is_dir()
    expire(tmp_path,datetime(2026,10,19,tzinfo=timezone.utc),delete=True)
    assert not (tmp_path/"delivery").exists()
    assert (tmp_path/"assets").is_dir() and (tmp_path/"unowned").is_dir()


def test_package_keeps_native_masks_and_smpl_parameters_without_dense_mesh(tmp_path):
    pytest.importorskip("pyarrow")
    import gzip,json
    import pyarrow.parquet as pq
    from arenalab_workbench.scene_assets.package import pack
    from arenalab_workbench.scene_assets.formats import write_json,sha256
    work=tmp_path/"work";work.mkdir();(work/"masks").mkdir()
    (work/"video.mp4").write_bytes(b"fixture-only: media is not decoded by pack")
    write_json(work/"manifest.json",dict(source=dict(video=dict(path=str(work/"video.mp4")))))
    np.savez(work/"frames.npz",ids=[0,2])
    mask=np.zeros((2,12,16),dtype="uint16");mask[:,2:8,4:9]=1
    np.savez(work/"masks/object-01-masks.npz",masks=mask,frame_indices=[0,1])
    geometry=tmp_path/"geometry.npz"
    np.savez(geometry,poses=np.tile(np.eye(4),(2,1,1)),k=np.tile(np.eye(3),(2,1,1)),
             ids=[0,2],labels=np.zeros((2,3,4),dtype="uint16"))
    write_json(geometry.with_suffix(".json"),dict(test_fixture=True))
    splats=tmp_path/"splats";splats.mkdir();(splats/"static.spz").write_bytes(b"fixture")
    write_json(splats/"static.json",{})
    objects=tmp_path/"objects";objects.mkdir();write_json(objects/"objects.json",[])
    parameter=tmp_path/"prior.npz"
    angles=np.arange(330,dtype="float64").reshape(2,55,3)/1000
    np.savez(parameter,local_axis_angle=angles,root_translation=np.zeros((2,3)),betas=np.zeros(100),
             vertices=np.ones((2,100,3)),faces=np.zeros((5,3),dtype="uint32"))
    original=sha256(parameter)
    output=tmp_path/"assets"
    pack(work,geometry,splats,objects,output,[parameter])
    rows=pq.read_table(output/"parameters-000.parquet").to_pylist()
    assert {row["name"] for row in rows}=={"local_axis_angle","root_translation","betas"}
    row=next(row for row in rows if row["name"]=="local_axis_angle")
    restored=np.frombuffer(row["data"],dtype=row["dtype"]).reshape(row["shape"])
    np.testing.assert_array_equal(restored,angles)
    assert sha256(parameter)==original
    with gzip.open(output/"masks.rle.jsonl.gz","rt") as stream:
        records=[json.loads(line) for line in stream]
    np.testing.assert_array_equal(rle_decode(records[0]["rle"]),mask[0]>0)
    assert records[1]["source_frame"]==2
    assert not list(output.glob("*.npz")) and not (output/"cloud.bin").exists()


def test_disk_backed_npz_does_not_materialize_sequence_and_rejects_changed_source(tmp_path):
    import json
    from arenalab_workbench.scene_assets.store import NpzArrays
    source=tmp_path/"input.npz"
    values=np.arange(600,dtype="float32").reshape(10,20,3)
    np.savez_compressed(source,points=values)
    cache=NpzArrays(source)
    assert isinstance(cache["points"],np.memmap)
    np.testing.assert_array_equal(cache["points"][4],values[4])
    # Container bind mounts may change absolute paths without changing bytes.
    receipt=cache.root/"source.json"
    metadata=json.loads(receipt.read_text());metadata["source"]="/another-mount/input.npz"
    receipt.write_text(json.dumps(metadata))
    NpzArrays(source)
    np.savez_compressed(source,points=values+1)
    with pytest.raises(ValueError,match="source changed"):
        NpzArrays(source)


def test_zarr_geometry_backprojects_one_frame_without_storing_dense_points(tmp_path):
    pytest.importorskip("zarr")
    from arenalab_workbench.scene_assets.store import create_geometry,GeometryStore
    root=create_geometry(tmp_path/"geometry.zarr",5,4,6)
    for i in range(5):
        root["ids"][i]=i*2;root["depth"][i]=2;root["valid"][i]=True
        root["k"][i]=np.array([[5,0,3],[0,5,2],[0,0,1.]])
        root["poses"][i]=np.eye(4)
    store=GeometryStore(tmp_path/"geometry.zarr")
    assert "points" not in store
    assert store.frame(2)["points"].shape==(4,6,3)
    np.testing.assert_array_equal(store.frame(2)["points"][...,2],2)


def test_sam_scale_bakes_only_geometry_and_preserves_anisotropic_axes():
    from arenalab_workbench.scene_assets.objects import metric_mesh
    vertices=np.eye(3)
    np.testing.assert_array_equal(metric_mesh(vertices,[2,3,4]),np.diag([2,3,4]))
    np.testing.assert_array_equal(vertices,np.eye(3))
    with pytest.raises(ValueError):metric_mesh(vertices,[1,-1,1])


def test_object_pose_interpolation_does_not_fill_occlusions_or_extrapolate():
    from arenalab_workbench.scene_assets.delivery import pose_at
    a=np.eye(4);b=np.eye(4);b[0,3]=2;b[:3,:3]=Rotation.from_euler("z",90,degrees=True).as_matrix()
    rows={0:a.ravel().tolist(),2:b.ravel().tolist(),4:None,6:b.ravel().tolist()}
    pose=pose_at(rows,1)
    assert pose[0,3]==1
    np.testing.assert_allclose(Rotation.from_matrix(pose[:3,:3]).as_euler("xyz",degrees=True)[2],45)
    assert pose_at(rows,3) is None and pose_at(rows,4) is None and pose_at(rows,7) is None


def test_static_feature_loop_is_verified_with_dynamic_pixels_excluded():
    pytest.importorskip("cv2")
    from arenalab_workbench.scene_assets.loops import verified_loop
    rng=np.random.default_rng(18)
    h,w=150,200
    image=rng.integers(0,256,(h,w,3),dtype="uint8")
    depth=np.full((h,w),2.)
    k=np.array([[200,0,100],[0,200,75],[0,0,1.]])
    points=backproject(depth,k,np.eye(4))
    labels=np.zeros((h,w),dtype="uint16");labels[30:110,60:130]=65535
    left=dict(images=image,labels=labels,valid=np.ones((h,w),bool),depth=depth,points=points)
    right={**left,"points":points+np.array([.4,-.1,.2])}
    matched,audit=verified_loop(left,right)
    assert matched is not None and audit["accepted"]
    assert audit["rmse_m"]<1e-6


def test_reconstruction_masks_keep_hardware_excluded_across_overlapping_prompts(tmp_path):
    import json
    from arenalab_workbench.scene_assets.reconstruct import MaskFrames
    work=tmp_path/"work";(work/"masks").mkdir(parents=True)
    (work/"prompts.json").write_text(json.dumps([["human-hardware","gripper"],["object-1","tool"]]))
    human=np.zeros((2,8,10),dtype="uint16");human[:,1:5,1:5]=1
    obj=np.zeros_like(human);obj[:,3:7,3:8]=3
    for name,value in [("human-hardware",human),("object-1",obj)]:
        np.savez_compressed(work/"masks"/(name+"-masks.npz"),masks=value,frame_indices=[0,1],source_frame_indices=[0,2])
    masks=MaskFrames(work,np.array([0,2]),tmp_path/"scratch")
    frame=masks.frame(1)
    assert (frame[human[1]>0]==65535).all()
    assert (frame[5:7,5:8]==1).all()


def test_association_retains_sam_identities_instead_of_spawning_per_frame():
    from arenalab_workbench.scene_assets.association import associate
    class Store:
        def __len__(self):return 15
        def frame(self,index):
            labels=np.zeros((20,24),dtype="uint16")
            labels[4:12,4:10]=1;labels[4:12,11:17]=2
            k=np.array([[40,0,12],[0,40,10],[0,0,1.]])
            depth=np.full(labels.shape,2.)
            return dict(labels=labels,valid=np.ones_like(labels,bool),depth=depth,ids=index*2,
                images=np.full((*labels.shape,3),128,dtype="uint8"),points=backproject(depth,k,np.eye(4)))
    tracks=associate(Store())
    assert len(tracks)==2
    assert all(len(t["observations"])==15 for t in tracks)
    assert all(len(t["source_labels"])==1 for t in tracks)
