"""Deferred novel-view RGB/depth/cloud rendering from public asset formats.

Robot meshes and per-geometry poses are produced by the existing URDF ingest /
retarget pipeline. This renderer never invents a robot/world registration.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import subprocess

import numpy as np

from .formats import read_arrays, depth_mm, sparse_cloud, write_json, sha256


def pose_at(rows, frame):
    """Interpolate adjacent observed poses; never bridge a nullable observation."""
    from bisect import bisect_left
    from scipy.spatial.transform import Rotation,Slerp
    keys=sorted(rows)
    if frame in rows:
        return None if rows[frame] is None else np.asarray(rows[frame]).reshape(4,4)
    j=bisect_left(keys,frame)
    if j==0 or j==len(keys) or rows[keys[j-1]] is None or rows[keys[j]] is None:
        return None
    left,right=keys[j-1],keys[j]
    # Input geometry can be at 15 Hz, but no interpolation across dropped observations.
    stride=min(np.diff(keys)) if len(keys)>1 else 1
    if right-left>stride:
        return None
    a,b=np.asarray(rows[left]).reshape(4,4),np.asarray(rows[right]).reshape(4,4)
    value=np.eye(4);weight=(frame-left)/(right-left)
    value[:3,3]=a[:3,3]*(1-weight)+b[:3,3]*weight
    value[:3,:3]=Slerp([left,right],Rotation.from_matrix([a[:3,:3],b[:3,:3]]))([frame]).as_matrix()[0]
    return value


def render(assets,cameras,output,decoder,robot_model=None,playback=None,robot_to_world=None,urdf=None,
           days=7,points=4096,robot_binding=None,geometry_fps=15):
    os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    import imageio.v2 as imageio
    import pyarrow.parquet as pq
    import pyrender
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation
    import torch
    import torch.nn.functional as F
    import trimesh
    import zarr
    from numcodecs import Blosc
    from gsplat import rasterization
    from . import POLICY

    if not 7<=days<=30 or points not in {1024,4096,8192} or geometry_fps not in {15,30}:
        raise ValueError("Delivery needs 7-30 day retention and a supported point budget")
    assets,output,decoder=map(Path,[assets,output,decoder])
    if output.exists():
        raise FileExistsError("Deliveries are immutable; choose a new output directory")
    manifest=json.loads((assets/"manifest.json").read_text())
    for path,record in manifest["assets"].items():
        if sha256(assets/path)!=record["sha256"]:
            raise ValueError(f"Permanent asset changed: {path}")
    camera=read_arrays(cameras)
    table=pq.read_table(cameras)
    ids=np.asarray(table["source_frame"].to_pylist())
    if len(ids)>1 and not np.all(np.diff(ids)==ids[1]-ids[0]):
        raise ValueError("RGB delivery requires a regular source clock")
    fps=30./(ids[1]-ids[0]) if len(ids)>1 else 30.
    if not all(k in camera for k in ["intrinsic","camera_to_world","image_size"]):
        raise ValueError("Target camera Parquet needs intrinsic, camera_to_world and image_size [H,W]")
    sizes=camera["image_size"].astype(int)
    if np.unique(sizes,axis=0).shape[0]!=1:
        raise ValueError("One delivery video must have a constant image size")
    h,w=map(int,sizes[0])
    robot_inputs=[robot_model,playback,robot_to_world,urdf]
    if any(v is not None for v in robot_inputs) and not all(v is not None for v in robot_inputs):
        raise ValueError("Robot rendering requires URDF, native GLB/playback and declared robot_to_world transform")
    if robot_model:
        if robot_binding is None:
            raise ValueError("Provide native export binding provenance; never guess mesh/pose order or units")
        binding=json.loads(Path(robot_binding).read_text())
        for key,path in [("urdf",urdf),("model",robot_model),("playback",playback)]:
            if binding.get(key+"_sha256")!=sha256(path):
                raise ValueError("Robot binding checksum mismatch: "+key)
        if binding.get("positions_unit")!="m" or not binding.get("source_world_unchanged"):
            raise ValueError("Robot binding must declare metre poses in the unchanged source world")
        reconstruction=json.loads((assets/"reconstruction.json").read_text())
        if reconstruction.get("world")=="estimated" and binding.get("task_coordinate_frame")!="estimated_reconstruction":
            raise ValueError("Estimated scene gauge is not a calibrated source-world alignment; supply the correct common frame")
    output.mkdir(parents=True)
    (output/"extensions/arenalabs").mkdir(parents=True)
    import shutil
    shutil.copyfile(cameras,output/"extensions/arenalabs/target-cameras.parquet")
    created=datetime.now(timezone.utc)
    record=dict(schema=POLICY,status="rendering",created_at=created.isoformat(),
                expires_at=(created+timedelta(days=days)).isoformat(),owned_temporary_delivery=True,
                source_asset_sha256=sha256(assets/"manifest.json"),camera_sha256=sha256(cameras),
                robot_urdf_sha256=sha256(urdf) if urdf else None,contact_generated=False)
    write_json(output/"delivery.json",record)
    scratch=output/"decoded.ply"
    subprocess.run([str(decoder),str(assets/"static.spz"),str(scratch)],check=True)
    splats=PlyData.read(str(scratch))["vertex"].data
    def stack(names):
        return torch.tensor(np.stack([splats[n] for n in names],axis=-1),device="cuda",dtype=torch.float32)
    means=stack(["x","y","z"])
    scales=stack(["scale_0","scale_1","scale_2"]).exp()
    quats=F.normalize(stack(["rot_0","rot_1","rot_2","rot_3"]),dim=-1)
    opacity=stack(["opacity"])[:,0].sigmoid()
    colors=(stack(["f_dc_0","f_dc_1","f_dc_2"])*.28209479177387814+.5).clamp(0,1)
    scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.7]*3)
    object_nodes=[]
    for folder in sorted((assets/"objects").iterdir()):
        if not folder.is_dir() or not (folder/"object.glb").exists():
            continue
        if not (folder/"trajectory.parquet").exists():
            raise ValueError("Generated object has no tracked trajectory: "+folder.name)
        mesh=trimesh.load(folder/"object.glb",force="scene").to_geometry()
        node=scene.add(pyrender.Mesh.from_trimesh(mesh))
        trajectories={int(row["source_frame"]):row["pose"] for row in pq.read_table(folder/"trajectory.parquet").to_pylist()}
        object_nodes.append((int(folder.name),node,trajectories))
    robot_nodes=[]
    if robot_model:
        robot=trimesh.load(robot_model,force="scene")
        motion=json.loads(Path(playback).read_text())
        world=np.asarray(json.loads(Path(robot_to_world).read_text()),dtype=float)
        if (world.shape!=(4,4) or not np.allclose(world[:3,:3].T@world[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(world[:3,:3]),1,atol=1e-6) or not np.allclose(world[3],[0,0,0,1])):
            raise ValueError("robot_to_world must be a declared rigid representation/calibration transform")
        translations=np.asarray(motion["positions"])
        rotations=Rotation.from_quat(np.asarray(motion["quaternions"]).reshape(-1,4)).as_matrix().reshape(*translations.shape[:2],3,3)
        geometry_names=binding.get("geometry_names")
        if geometry_names is None or len(geometry_names)!=translations.shape[1] or set(geometry_names)!=set(robot.geometry):
            raise ValueError("Native mesh names and pose columns disagree")
        for i in range(translations.shape[1]):
            node=scene.add(pyrender.Mesh.from_trimesh(robot.geometry[geometry_names[i]]))
            robot_nodes.append((i,node))
    camera_node=scene.add(pyrender.IntrinsicsCamera(1,1,0,0,znear=.01,zfar=65))
    light=scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.5))
    renderer=pyrender.OffscreenRenderer(w,h)
    arrays=zarr.open_group(str(output/"depth-and-points.zarr"),mode="w")
    compressor=Blosc(cname="zstd",clevel=5,shuffle=Blosc.BITSHUFFLE)
    n=len(ids)
    geometry_step=max(1,round(fps/geometry_fps))
    count_geometry=len(range(0,n,geometry_step))
    arrays.create_dataset("source_frame",data=ids[::geometry_step],compressor=compressor)
    arrays.create_dataset("timestamp_s",data=ids[::geometry_step]/30,compressor=compressor)
    depth_out=arrays.create_dataset("depth_mm",shape=(count_geometry,h,w),chunks=(1,h,w),dtype="u2",compressor=compressor)
    observed=arrays.create_dataset("render_valid",shape=(count_geometry,h,w),chunks=(1,h,w),dtype="bool",compressor=compressor)
    segmentation=arrays.create_dataset("instance_id",shape=(count_geometry,h,w),chunks=(1,h,w),dtype="u2",compressor=compressor)
    origin=arrays.create_dataset("surface_origin",shape=(count_geometry,h,w),chunks=(1,h,w),dtype="u1",compressor=compressor)
    xyz=arrays.create_dataset("points_xyz",shape=(count_geometry,points,3),chunks=(1,points,3),dtype="f4",compressor=compressor)
    rgb=arrays.create_dataset("points_rgb",shape=(count_geometry,points,3),chunks=(1,points,3),dtype="u1",compressor=compressor)
    point_valid=arrays.create_dataset("points_valid",shape=(count_geometry,points),chunks=(1,points),dtype="bool",compressor=compressor)
    point_labels=arrays.create_dataset("points_instance_id",shape=(count_geometry,points),chunks=(1,points),dtype="u2",compressor=compressor)
    video=output/"videos/observation.images.robot/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    writer=imageio.get_writer(video,fps=float(fps),codec="libx264",macro_block_size=1)
    try:
        for i,frame in enumerate(ids):
            k=camera["intrinsic"][i];pose=camera["camera_to_world"][i]
            hidden=[]
            for _,node,trajectory in object_nodes:
                value=pose_at(trajectory,int(frame))
                if value is None:
                    scene.remove_node(node);hidden.append(node)
                else:
                    scene.set_pose(node,value)
            for j,node in robot_nodes:
                if frame>=len(translations):
                    raise ValueError("Robot motion ends before target camera trajectory")
                transform=np.eye(4);transform[:3,:3]=rotations[frame,j];transform[:3,3]=translations[frame,j]
                scene.set_pose(node,world@transform)
            camera_node.camera=pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2],znear=.01,zfar=65)
            scene.set_pose(camera_node,pose@np.diag([1,-1,-1,1]))
            scene.set_pose(light,pose@np.diag([1,-1,-1,1]))
            mesh_rgb,mesh_z=renderer.render(scene)
            node_labels={node:[label&255,(label>>8)&255,0] for label,node,_ in object_nodes if node not in hidden}
            node_labels.update({node:[254,255,0] for _,node in robot_nodes})
            mesh_ids,_=renderer.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map=node_labels)
            mesh_ids=mesh_ids[...,0].astype("uint16")+256*mesh_ids[...,1].astype("uint16")
            for node in hidden:
                scene.add_node(node)
            with torch.inference_mode():
                rendered,alpha,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacity,
                    colors=colors,viewmats=torch.tensor(np.linalg.inv(pose)[None],device="cuda",dtype=torch.float32),
                    Ks=torch.tensor(k[None],device="cuda",dtype=torch.float32),width=w,height=h,render_mode="RGB+ED")
            background=rendered[0].cpu().numpy()
            z=background[...,3]
            valid=(alpha[0,...,0].cpu().numpy()>.5)&np.isfinite(z)&(z>0)
            visible_mesh=(mesh_z>0)&(~valid|(mesh_z<z))
            image=np.clip(background[...,:3]*255,0,255).astype("uint8")
            image[visible_mesh]=mesh_rgb[visible_mesh,:3]
            z[visible_mesh]=mesh_z[visible_mesh];valid|=visible_mesh
            z[~valid]=0
            encoded=depth_mm(z)
            semantic=np.zeros((h,w),dtype="uint16")
            semantic[visible_mesh]=mesh_ids[visible_mesh]
            semantic[encoded==0]=65535
            provenance=np.where(encoded>0,1,0).astype("uint8")
            provenance[visible_mesh]=2
            provenance[semantic==65534]=3
            provenance[encoded==0]=0
            if i%geometry_step==0:
                g=i//geometry_step
                depth_out[g]=encoded;observed[g]=encoded>0
                segmentation[g]=semantic;origin[g]=provenance
                cloud=sparse_cloud(np.where(encoded>0,z,0),k,pose,points,image,seed=int(frame),labels=semantic)
                count=len(cloud["xyz"])
                xyz[g,:count]=cloud["xyz"];rgb[g,:count]=cloud["rgb"];point_valid[g,:count]=True
                point_labels[g,:count]=cloud["instance_ids"]
            writer.append_data(image)
    finally:
        writer.close();renderer.delete()
    scratch.unlink()  # Only this function's decoded temporary PLY, never source assets.
    arrays.attrs.update(depth_units="millimetres",invalid_depth=0,cloud_frame="source world",fps=float(fps/geometry_step),rgb_fps=float(fps),
                        instance_semantics="0 static background, object ids from assets, 65534 robot, 65535 invalid",
                        depth_semantics="generated view; gaussian expected Z-depth plus mesh z-buffer; not measured depth")
    arrays.attrs["surface_origin_codes"]={"0":"invalid/unknown","1":"static GS fitted to observations","2":"SAM 3D generated object","3":"target robot CAD"}
    info=dict(codebase_version="v3.0",robot_type="target" if urdf else "scene-only",fps=float(fps),total_episodes=1,
              total_frames=n,total_videos=1,total_tasks=1,
              video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
              features={"observation.images.robot":dict(dtype="video",shape=[h,w,3],names=["height","width","channels"])})
    write_json(output/"meta/info.json",info)
    complete=False
    if robot_model:
        from .delivery_join import finalize
        finalize(output,assets,playback,ids)
        complete=True
    record.update(status="complete_lerobot_episode" if complete else "rendered_scene_modalities",frames=n,fps=float(fps),complete_lerobot_episode=complete,
                  geometry_frames=count_geometry,geometry_fps=float(fps/geometry_step),
                  action_export="Native retargeted playback joined by unchanged source frame" if complete else "No robot episode requested",
                  total_bytes=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                  provenance="novel-view synthetic RGB/depth; observed coverage not certified",
                  robot_mesh_pose_binding_verified=True if robot_model else None,
                  binding_provenance=binding if robot_model else None)
    write_json(output/"delivery.json",record)
