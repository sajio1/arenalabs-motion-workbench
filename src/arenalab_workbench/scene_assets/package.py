"""Robot-independent permanent assets, plus explicit temporary delivery requests."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import shutil
import gzip

import numpy as np

from . import POLICY
from .formats import write_json, sha256, rle_encode, write_arrays, storage_budget
from .store import GeometryStore, NpzArrays


def pack(work, geometry_path, splats, objects, destination, parameters=()):
    work, destination = Path(work), Path(destination)
    if destination.exists():
        raise FileExistsError("An asset package is immutable; use a new destination")
    geometry=GeometryStore(geometry_path)
    splats, objects = Path(splats), Path(objects)
    required = [splats/"static.spz",splats/"static.json",objects/"objects.json"]
    if not all(path.is_file() for path in required):
        raise ValueError("Geometry, static splats and object reconstruction must finish before packaging")
    object_status=objects/"status.json"
    if object_status.exists() and not json.loads(object_status.read_text()).get("complete"):
        raise ValueError("Object job is incomplete or diagnostically limited; do not package it as the full workflow")
    destination.mkdir(parents=True)
    try:
        image_size=np.tile(geometry["labels"].shape[1:],(len(geometry),1))
        write_arrays(destination/"cameras.parquet",dict(camera_to_world=np.asarray(geometry["poses"]),
                     intrinsic=np.asarray(geometry["k"]),image_size=image_size),np.asarray(geometry["ids"]))
        if (work/"cameras.npz").exists():
            with np.load(work/"cameras.npz") as source:
                n=len(source["poses"])
                prep=json.loads((work/"preparation.json").read_text())
                write_arrays(destination/"input-cameras.parquet",dict(camera_to_world=source["poses"],
                             intrinsic=np.tile(source["intrinsic"],(n,1,1)),
                             image_size=np.tile([prep["height"],prep["width"]],(n,1))))
        for name in ["static.spz","static.json"]:
            shutil.copyfile(splats/name,destination/name)
        for name in ["semantic.json","prompts.json","preparation.json"]:
            if (work/name).exists():
                shutil.copyfile(work/name,destination/name)
        # Raw model tensors/debug images stay in the immutable research job,
        # not the compact permanent asset package.
        (destination/"objects").mkdir()
        for name in ["objects.json","association.json","status.json"]:
            if (objects/name).exists():
                shutil.copyfile(objects/name,destination/"objects"/name)
        for folder in objects.iterdir():
            if not folder.is_dir():
                continue
            for name in ["object.glb","trajectory.parquet","object.json"]:
                if (folder/name).exists():
                    target=destination/"objects"/folder.name/name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(folder/name,target)
        with gzip.open(destination/"geometry-labels.rle.jsonl.gz","wt",encoding="utf-8") as stream:
            for index in range(len(geometry)):
                frame,labels=geometry["ids"][index],np.asarray(geometry["labels"][index])
                for label in np.unique(labels):
                    if label:
                        stream.write(json.dumps(dict(source_frame=int(frame),timestamp_s=float(frame/30),
                            instance_id=int(label),rle=rle_encode(labels==label)))+"\n")
        # Keep the original SAM masks too: downsampled geometry labels must not
        # silently replace native segmentation observations in permanent storage.
        source_ids=np.load(work/"frames.npz")["ids"]
        with gzip.open(destination/"masks.rle.jsonl.gz","wt",encoding="utf-8") as stream:
            for mask_path in sorted((work/"masks").glob("*-masks.npz")):
                saved=NpzArrays(mask_path,Path(geometry_path).parent/"pack-input-arrays"/mask_path.stem)
                for index,mask in zip(saved["frame_indices"],saved["masks"]):
                        frame=int(source_ids[int(index)])
                        for instance in np.unique(mask):
                            if instance:
                                stream.write(json.dumps(dict(source_frame=frame,timestamp_s=frame/30,
                                    mask_key=mask_path.name.removesuffix("-masks.npz"),
                                    source_instance=int(instance),rle=rle_encode(mask==instance)))+"\n")
        # Original input is evidence and must outlive the temporary delivery. Keep
        # one native video here; do not duplicate it once per target embodiment.
        manifest = json.loads((work/"manifest.json").read_text())
        source_video = Path(manifest["source"]["video"]["path"])
        shutil.copyfile(source_video,destination/"original.mp4")
        evidence = dict(video_sha256=sha256(source_video),
                        prepared_manifest_sha256=sha256(work/"manifest.json"),
                        source_clock=manifest["source"]["video"].get("frame_times"),
                        source_start_seconds=manifest["source"]["video"].get("start_seconds",0))
        if evidence["source_clock"] is not None:
            clock=np.asarray(evidence.pop("source_clock"))
            write_arrays(destination/"source-clock.parquet",dict(source_time_s=clock))
            evidence["source_clock"]="source-clock.parquet"
        # Preserve each parameter array exactly, including shape and original dtype.
        # Per-array rows avoid inventing frame axes for constant body shape/betas.
        import pyarrow as pa
        import pyarrow.parquet as pq
        for index, parameter in enumerate(parameters):
            rows=[]
            with np.load(parameter,allow_pickle=False) as saved:
                # SMPL-X pose parameters determine these derived arrays. Preserve
                # their immutable original file outside the canonical package,
                # rather than storing a full human mesh at every timestamp.
                is_smplx={"betas","local_axis_angle","root_translation"}.issubset(saved.files)
                omitted={"vertices","faces","joints","global_rotation"} if is_smplx else set()
                for name in saved.files:
                    if name in omitted:
                        continue
                    value=saved[name]
                    if value.dtype.hasobject:
                        raise ValueError("Object arrays cannot be archived as parameter tensors")
                    rows.append(dict(name=name,shape=list(value.shape),dtype=value.dtype.str,
                                     data=np.ascontiguousarray(value).tobytes()))
            table=pa.Table.from_pylist(rows)
            pq.write_table(table,destination/f"parameters-{index:03}.parquet",compression="zstd")
            evidence.setdefault("parameters",[]).append(dict(file=f"parameters-{index:03}.parquet",
                source_sha256=sha256(parameter),omitted_derived_fields=sorted(omitted.intersection(saved.files)),
                reconstruction="Same licensed SMPL-X model and shape conventions required" if is_smplx else "Stored input arrays"))
        shutil.copyfile(Path(geometry_path).with_suffix(".json"),destination/"reconstruction.json")
        assets={p.relative_to(destination).as_posix():dict(bytes=p.stat().st_size,sha256=sha256(p))
                for p in destination.rglob("*") if p.is_file()}
        write_json(destination/"manifest.json",dict(schema=POLICY,experimental=True,
                   robot_independent=True,frames=len(geometry),source_fps=30,
                   geometry_fps="source frame ids in cameras.parquet",per_frame_clouds_stored=False,
                   mask_pixel_space="Original SAM/prepared RGB dimensions, not automatically native video dimensions",
                   source=evidence,assets=assets,total_bytes=sum(v["bytes"] for v in assets.values()),
                   parameters_supplied=len(parameters),motion_completeness_verified=False,
                   object_surface_policy="SAM 3D complete mesh hypothesis, FoundationPose trajectory; generated backside distinct from observed evidence",
                   quality_status="Computed assets; geometry, identity and task accuracy are not certified by successful execution",
                   scratch_policy="immutable raw model chunks retained outside permanent package until validation"))
    except Exception as error:
        write_json(destination/"failed.json",dict(error=str(error),complete=False))
        raise


def delivery_request(assets, urdf, cameras, output, days=7, points=4096):
    """Describe deferred rendering; a request is never reported as rendered data."""
    if not 7 <= days <= 30:
        raise ValueError("Delivery retention must be 7-30 days")
    if points not in {1024,4096,8192}:
        raise ValueError("Point budget must be 1024, 4096 or 8192")
    assets, urdf, cameras = Path(assets), Path(urdf), Path(cameras)
    for path in [assets/"manifest.json",urdf,cameras]:
        if not path.is_file():
            raise ValueError(f"Required delivery input missing: {path}")
    created=datetime.now(timezone.utc)
    request=dict(schema=POLICY,status="awaiting_renderer",created_at=created.isoformat(),
                 expires_at=(created+timedelta(days=days)).isoformat(),
                 asset_manifest_sha256=sha256(assets/"manifest.json"),
                 urdf_sha256=sha256(urdf),camera_trajectory_sha256=sha256(cameras),
                 formats=dict(rgb_and_actions="LeRobot v3",depth="Zarr uint16 millimetres; 0 invalid; validity mask required",
                              points=dict(count=points,source="rendered depth",default_fps=15,uncompressed_gb_per_hour=storage_budget(points,fps=15))),
                 renderer_requirements=["declared source-world camera/robot transforms", "URDF mesh checksums",
                    "GS background depth composed with object and robot mesh z-buffer",
                    "coverage and observed/generated provenance masks", "no human-to-robot scene registration"],
                 retention="Only this delivery expires. Original evidence and scene assets are retained.",
                 automatic_deletion_enabled=False)
    write_json(output,request)
    return request
