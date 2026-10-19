"""SAM 3D complete object hypotheses + FoundationPose 6DoF tracking.

Weights are provisioned explicitly. There is no TSDF, ICP, hand attachment or
primitive fallback. Generated hidden geometry is identified as a hypothesis.
"""
from pathlib import Path
import gc
import json
import os
import sys
import time
import numpy as np

from .association import associate
from .formats import sha256, write_json
from .object_frames import ObjectFrames


def metric_mesh(vertices, scale):
    """Bake predicted metric scale in SAM-native axes, leaving rigid pose to FP."""
    scale = np.asarray(scale).reshape(-1)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("SAM 3D must provide three finite positive metric scales")
    return np.asarray(vertices) * scale


def compact_mesh(mesh,face_budget=20000):
    """Official visibility-aware simplification / boundary repair in a unit box.

Plain quadric decimation introduced holes in closed SAM meshes. Preserve the
raw prior, remove invisible components and repair boundaries with upstream's
complete postprocessor instead. Coordinates are restored to the SAME frame.
"""
    if len(mesh.faces)<=face_budget:
        return mesh
    os.environ.setdefault("LIDRA_SKIP_INIT","true")
    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import postprocess_mesh
    from scipy.spatial import cKDTree
    import trimesh
    center=(mesh.bounds[0]+mesh.bounds[1])/2
    size=float(np.max(mesh.extents))
    vertices,faces=postprocess_mesh(((mesh.vertices-center)/size).astype("float32"),mesh.faces,
        simplify=True,simplify_ratio=.95,fill_holes=True,fill_holes_max_hole_size=.04,
        fill_holes_max_hole_nbe=56,fill_holes_resolution=1024,fill_holes_num_views=1000,verbose=False)
    reduced=trimesh.Trimesh(vertices=vertices*size+center,faces=faces,process=False)
    nearest=cKDTree(mesh.vertices).query(reduced.vertices,k=1)[1]
    reduced.visual.vertex_colors=np.asarray(mesh.visual.vertex_colors)[nearest]
    return reduced


class Sam3D:
    def __init__(self, source, config):
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        sys.path.insert(0, str(Path(source).resolve()))
        from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
        self.camera_rotation = camera_to_pytorch3d_camera().rotation[0].numpy()
        cfg = OmegaConf.load(config)
        cfg.workspace_dir = str(Path(config).resolve().parent)
        cfg.rendering_engine = "pytorch3d"
        cfg.compile_model = False
        # External MapAnything pointmaps replace MoGe; avoid another network and
        # its implicit weight download. This is an upstream-supported input.
        cfg.depth_model = None
        self.pipeline = instantiate(cfg)

    def generate(self, frame, mask):
        import torch
        depth, k = frame["depth"], frame["k"]
        y,x = np.indices(depth.shape)
        camera = (np.stack([x,y,np.ones_like(x)],axis=-1) @ np.linalg.inv(k).T) * depth[...,None]
        pointmap = camera @ self.camera_rotation
        pointmap[~frame["valid"]] = np.nan
        rgba = np.concatenate([frame["images"],(mask.astype("uint8")*255)[...,None]],axis=-1)
        with torch.inference_mode():
            return self.pipeline.run(rgba, None, seed=42, stage1_only=False,
                with_mesh_postprocess=False, with_texture_baking=False,
                with_layout_postprocess=False, use_vertex_color=True,
                pointmap=torch.as_tensor(pointmap,device="cuda",dtype=torch.float32))


def tracking_rows(store, observations, estimator, surface_tree=None):
    """Track from the best observed seed in both directions, retaining gaps."""
    lookup = {row["index"]: row for row in observations}
    seed = max(observations,key=lambda x:x["area"])["index"]
    rows,seed_state = {},None
    for direction, indices in [(1,range(seed,len(store))),(-1,range(seed-1,-1,-1))]:
        previous = None
        if direction == -1:
            estimator.pose_last = seed_state.clone()
            previous = seed
        for index in indices:
            frame = store.frame(index)
            source_id = int(frame["ids"])
            row = dict(source_frame=source_id,timestamp_s=source_id/30,visible=index in lookup,
                       pose=None,status="unobserved")
            if index not in lookup:
                rows[index] = row
                continue
            mask = (frame["labels"] == lookup[index]["label"]) & frame["valid"]
            depth = np.where(frame["valid"] & (frame["labels"] != 65535),frame["depth"],0).astype("float32")
            # Match upstream's float64 calibration contract. Its crop-radius
            # construction uses numpy float64; float32 K produces mixed matmul.
            k = frame["k"].astype("float64")
            try:
                if previous is None or abs(source_id-int(store["ids"][previous])) > 6:
                    pose = estimator.register(K=k,rgb=frame["images"],depth=depth,
                                              ob_mask=mask,iteration=5)
                    row["status"] = "foundationpose_registered"
                else:
                    pose = estimator.track_one(rgb=frame["images"],depth=depth,K=k,iteration=2)
                    row["status"] = "foundationpose_tracked"
                if not np.isfinite(pose).all():
                    raise ValueError("Nonfinite object pose")
                world_pose=frame["poses"] @ pose
                row["pose"] = world_pose.reshape(-1).tolist()
                if surface_tree is not None:
                    observed=frame["points"][mask]
                    observed=observed[np.linspace(0,len(observed)-1,min(1024,len(observed))).astype(int)]
                    local=(observed-world_pose[:3,3]) @ world_pose[:3,:3]
                    distances=surface_tree.query(local,k=1)[0]
                    row.update(surface_median_m=float(np.median(distances)),surface_p95_m=float(np.quantile(distances,.95)))
                if index == seed:
                    seed_state = estimator.pose_last.clone()
                previous = index
            except Exception as error:
                row.update(status="tracking_failed",error=type(error).__name__+": "+str(error))
                previous = None
                if index == seed:
                    raise RuntimeError("FoundationPose seed registration failed: "+str(error)) from error
            rows[index] = row
    return [rows[i] for i in range(len(store))]


def run(geometry_path, output, sam_source, sam_config, foundation_source, work, limit_tracks=None):
    import torch
    import trimesh
    if not torch.cuda.is_available():
        raise RuntimeError("Object generation/tracking requires a cloud CUDA worker")
    for path in [sam_source,sam_config,foundation_source]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Provision the model explicitly before inference: {path}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Object priors are immutable; use a new output directory")
    output.mkdir(parents=True,exist_ok=True)
    store = ObjectFrames(geometry_path,work)
    tracks = associate(store)
    total_tracks = len(tracks)
    write_json(output/"association.json",dict(tracks=tracks,policy="geometry, appearance, one-to-one matching; ambiguous identities separate"))
    if limit_tracks is not None:
        tracks = sorted(tracks,key=lambda t:max(r["area"] for r in t["observations"]),reverse=True)[:limit_tracks]
    generator = Sam3D(sam_source,sam_config)
    results = []
    for track in tracks:
        folder = output/str(track["id"])
        folder.mkdir()
        started = time.monotonic()
        record = dict(id=track["id"],source_labels=track["source_labels"],keyframe=track["keyframe"],
                      backside="SAM 3D generated; not an observed surface",status="generating")
        try:
            frame = store.frame(track["keyframe"])
            seed = next(o for o in track["observations"] if o["index"] == track["keyframe"])
            predicted = generator.generate(frame,(frame["labels"] == seed["label"]) & frame["valid"])
            mesh = predicted["mesh"][0]
            vertices = mesh.vertices.detach().float().cpu().numpy()
            faces = mesh.faces.detach().cpu().numpy()
            colors = mesh.vertex_attrs[:,:3].detach().float().cpu().numpy()
            tensors = {name:predicted[name].detach().float().cpu().numpy() for name in ["scale","rotation","translation"]}
            np.savez_compressed(folder/"sam3d-prior.npz",vertices=vertices,faces=faces,colors=colors,**tensors)
            canonical = trimesh.Trimesh(vertices=metric_mesh(vertices,tensors["scale"]),faces=faces,
                vertex_colors=np.rint(np.clip(colors,0,1)*255).astype("uint8"),process=False)
            canonical = compact_mesh(canonical)
            canonical.export(folder/"object.glb")
            record.update(status="mesh_ready",raw_triangles=len(faces),triangles=len(canonical.faces),sam_seconds=time.monotonic()-started,
                raw_prior_sha256=sha256(folder/"sam3d-prior.npz"),scale=tensors["scale"].tolist(),
                mesh_axes="SAM-native; scale baked, rigid pose estimated separately",
                metric_source="MapAnything pointmap supplied to SAM 3D layout decoder")
            del predicted,mesh
        except Exception as error:
            record.update(status="generation_failed",error=type(error).__name__+": "+str(error))
        write_json(folder/"object.json",record)
        results.append(record)
        print(json.dumps(record),flush=True)
    del generator
    gc.collect();torch.cuda.empty_cache()
    track_generated(store,output,tracks,results,foundation_source,total_tracks,limit_tracks)


def track_generated(store,output,tracks,results,foundation_source,total_tracks,limit_tracks=None):
    import trimesh
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree
    sys.path.insert(0,str(Path(foundation_source).resolve()))
    from estimater import FoundationPose
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    scorer,refiner,context = ScorePredictor(),PoseRefinePredictor(),dr.RasterizeCudaContext()
    by_id = {t["id"]:t for t in tracks}
    for record in results:
        if record["status"] != "mesh_ready":
            continue
        folder = output/str(record["id"])
        started = time.monotonic()
        try:
            mesh = trimesh.load(folder/"object.glb",force="mesh",process=False)
            estimator = FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,
                mesh=mesh,scorer=scorer,refiner=refiner,glctx=context,debug=0,debug_dir=str(folder/"tracker-debug"))
            surface=trimesh.sample.sample_surface(mesh,20000,seed=0)[0]
            rows = tracking_rows(store,by_id[record["id"]]["observations"],estimator,cKDTree(surface))
            schema = pa.schema([("source_frame",pa.int64()),("timestamp_s",pa.float64()),("visible",pa.bool_()),
                                ("pose",pa.list_(pa.float64())),("status",pa.string()),("error",pa.string()),
                                ("surface_median_m",pa.float64()),("surface_p95_m",pa.float64())])
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),folder/"trajectory.parquet",compression="zstd")
            record.update(status="generated_mesh_and_tracked_pose",tracked_frames=sum(r["pose"] is not None for r in rows),
                          frames=len(rows),tracking_seconds=time.monotonic()-started,tracking="FoundationPose pretrained registration/refinement")
            errors=[r["surface_median_m"] for r in rows if "surface_median_m" in r]
            record.update(observed_surface_median_mm=float(np.median(errors))*1000 if errors else None,
                          diagnostic_semantics="Observed MapAnything mask points to sampled asset surface; not pose ground truth or task success")
        except Exception as error:
            record.update(status="tracking_failed",error=type(error).__name__+": "+str(error))
        write_json(folder/"object.json",record)
        write_json(output/"objects.json",results)
        print(json.dumps(record),flush=True)
    write_json(output/"objects.json",results)
    write_json(output/"status.json",dict(complete=limit_tracks is None and len(tracks)==total_tracks and all(r["status"] == "generated_mesh_and_tracked_pose" for r in results),
        total_tracks=total_tracks,processed_tracks=len(tracks),diagnostic_limit=limit_tracks,
        backend="SAM 3D Objects + FoundationPose",source_geometry=str(store.path)))
    if any(r["status"] != "generated_mesh_and_tracked_pose" for r in results):
        raise RuntimeError("Object assets failed; see retained per-object receipts")


def retrack(geometry,work,priors,output,foundation_source):
    """Retry tracking into a fresh directory without rerunning or altering SAM."""
    import shutil
    import trimesh
    priors,output=Path(priors),Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh tracking output directory")
    output.mkdir(parents=True)
    association=json.loads((priors/"association.json").read_text())
    results=[]
    for folder in sorted(priors.iterdir()):
        if not folder.is_dir() or not (folder/"object.glb").exists():
            continue
        target=output/folder.name;target.mkdir()
        original=trimesh.load(folder/"object.glb",force="mesh",process=False)
        compact=compact_mesh(original)
        compact.export(target/"object.glb")
        record=json.loads((folder/"object.json").read_text())
        record.update(status="mesh_ready",raw_prior_directory=str(folder),mesh_sha256=sha256(folder/"object.glb"))
        record.update(raw_triangles=len(original.faces),triangles=len(compact.faces),asset_policy="Official visibility-aware mesh postprocess; original frame restored; raw geometry retained")
        record.pop("error",None)
        results.append(record)
    tracks=[t for t in association["tracks"] if any(r["id"]==t["id"] for r in results)]
    write_json(output/"association.json",association)
    track_generated(ObjectFrames(geometry,work),output,tracks,results,foundation_source,len(association["tracks"]),
                    None if len(tracks)==len(association["tracks"]) else len(tracks))


def postprocess_existing(priors,output):
    """Improve assets without rerunning SAM or changing any saved 6DoF value."""
    import shutil
    import trimesh
    priors,output=Path(priors),Path(output)
    if output.exists():
        raise FileExistsError("Postprocessing needs a fresh output directory")
    original_status=json.loads((priors/"status.json").read_text())
    if not original_status.get("complete"):
        raise ValueError("Finish the original learned object run before postprocessing")
    output.mkdir(parents=True)
    shutil.copyfile(priors/"association.json",output/"association.json")
    records=[]
    for record in json.loads((priors/"objects.json").read_text()):
        source=priors/str(record["id"]);target=output/str(record["id"]);target.mkdir()
        started=time.monotonic()
        original=trimesh.load(source/"object.glb",force="mesh",process=False)
        mesh=compact_mesh(original)
        mesh.export(target/"object.glb")
        shutil.copyfile(source/"trajectory.parquet",target/"trajectory.parquet")
        record.update(raw_mesh_sha256=sha256(source/"object.glb"),raw_trajectory_sha256=sha256(source/"trajectory.parquet"),
            pose_values_unchanged=True,raw_triangles=len(original.faces),triangles=len(mesh.faces),
            watertight=bool(mesh.is_watertight),mesh_postprocess_seconds=time.monotonic()-started,
            asset_policy="Official visibility-aware simplification/boundary repair; same object coordinates")
        write_json(target/"object.json",record);records.append(record)
        print(json.dumps(record),flush=True)
    write_json(output/"objects.json",records)
    write_json(output/"status.json",{**original_status,"postprocess":"Official SAM 3D mesh postprocessor; raw poses retained"})
