"""Finalize a rendered asset delivery as a complete, time-aligned LeRobot v3 episode.

Reuse the current exporter's video normalization/statistics and table schema.
No motion model, optimizer, contacts or private format decoder is run here.
"""
from pathlib import Path
import json
import shutil
import numpy as np

from .formats import write_json,sha256


def finalize(output, assets, playback, source_ids):
    import pyarrow as pa
    import pandas as pd
    from arenalab_workbench.dataset_io.export import parquet,stats,normalize_video,video_stats
    output,assets,playback = map(Path,[output,assets,playback])
    motion = json.loads(playback.read_text())
    source_ids = np.asarray(source_ids)
    if not np.array_equal(source_ids,np.arange(len(source_ids))) or float(motion["fps"])!=30:
        raise ValueError("Complete LeRobot delivery requires the complete contiguous original 30 Hz clock")
    state = np.asarray(motion["joints"]["values"],dtype="float32")
    names = motion["joints"]["names"]
    n = len(source_ids)
    if state.shape!=(n,len(names)) or not np.isfinite(state).all():
        raise ValueError("Native playback state does not cover the source clock")
    action = np.concatenate([state[1:],state[-1:]])
    arrays = {"observation.state":state,"action":action,"timestamp":np.arange(n,dtype="float32")/30,
        "frame_index":source_ids.astype("int64"),"episode_index":np.zeros(n,dtype="int64"),
        "index":source_ids.astype("int64"),"task_index":np.zeros(n,dtype="int64")}
    columns = {key:pa.array(value.tolist(),type=pa.list_(pa.float32(),value.shape[1])) if value.ndim==2 else pa.array(value)
               for key,value in arrays.items()}
    parquet(output/"data/chunk-000/file-000.parquet",columns)
    features = {key:dict(dtype=str(v.dtype),shape=[v.shape[1]] if v.ndim==2 else [1],names=names if v.ndim==2 else None)
                for key,v in arrays.items()}
    task = "Retargeted manipulation"
    if (assets/"semantic.json").exists():
        semantics=json.loads((assets/"semantic.json").read_text())
        task=semantics.get("task") or semantics.get("instruction") or task
    pd.DataFrame({"task_index":[0]},index=[task]).to_parquet(output/"meta/tasks.parquet")
    episode = dict(episode_index=[0],tasks=[[task]],length=[n],dataset_from_index=[0],dataset_to_index=[n],
        **{"data/chunk_index":[0],"data/file_index":[0],"meta/episodes/chunk_index":[0],"meta/episodes/file_index":[0]})
    statistics = {key:stats(value) for key,value in arrays.items()}
    manifest = json.loads((assets/"manifest.json").read_text())
    clock_path = assets/"source-clock.parquet"
    if clock_path.exists():
        from .formats import read_arrays
        clock = read_arrays(clock_path)["source_time_s"]
    else:
        clock = None
    original = output/"videos/observation.images.original/chunk-000/file-000.mp4"
    normalize_video(assets/"original.mp4",original,n,manifest["source"].get("source_start_seconds",0),clock)
    for name in ["original","robot"]:
        key = "observation.images."+name
        video = output/f"videos/{key}/chunk-000/file-000.mp4"
        import cv2
        cap = cv2.VideoCapture(str(video))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h,w,fps = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if count!=n or abs(fps-30)>.01:
            raise ValueError(f"{key} clock does not match action table")
        features[key] = dict(dtype="video",shape=[h,w,3],names=["height","width","channels"],
            info={"video.fps":30,"video.codec":"h264","video.pix_fmt":"yuv420p","video.is_depth_map":False,"has_audio":False})
        statistics[key] = video_stats(video,n)
        for suffix,value in [("chunk_index",0),("file_index",0),("from_timestamp",0.),("to_timestamp",n/30)]:
            episode[f"videos/{key}/{suffix}"] = [value]
    for key,values in statistics.items():
        for name,value in values.items():
            episode[f"stats/{key}/{name}"] = [value]
    write_json(output/"meta/stats.json",statistics)
    parquet(output/"meta/episodes/chunk-000/file-000.parquet",episode)
    write_json(output/"meta/info.json",dict(codebase_version="v3.0",robot_type="custom",total_episodes=1,total_frames=n,
        total_tasks=1,total_videos=2,total_chunks=1,chunks_size=1000,data_files_size_in_mb=100,video_files_size_in_mb=200,
        fps=30,splits={"train":"0:1"},data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",features=features))
    extension=output/"extensions/arenalabs"
    extension.mkdir(parents=True,exist_ok=True)
    for name in ["masks.rle.jsonl.gz","geometry-labels.rle.jsonl.gz","semantic.json","prompts.json",
                 "preparation.json","cameras.parquet","input-cameras.parquet","source-clock.parquet"]:
        if (assets/name).exists():
            shutil.copyfile(assets/name,extension/name)
    for path in assets.glob("parameters-*.parquet"):
        shutil.copyfile(path,extension/path.name)
    shutil.copyfile(assets/"objects/association.json",extension/"object-association.json")
    write_json(output/"extensions/arenalabs/manifest.json",dict(schema="scene-asset-delivery-v1",fps=30,frames=n,
        source_asset_sha256=sha256(assets/"manifest.json"),native_playback_sha256=sha256(playback),
        depth_and_cloud="../../depth-and-points.zarr",depth_units="millimetres",contact_generated=False,
        action_semantics="Next-frame generated kinematic joint position; final frame repeats; not measured actuation",
        state_units=motion["joints"].get("units",motion["joints"].get("unit")),
        raw_observations="parameters-*.parquet, original SAM RLE and source calibration copied from the canonical package",
        geometry_clock="depth-and-points.zarr/source_frame and timestamp_s; not implicitly the RGB clock",
        scene_geometry="Generated novel view from static SPZ + SAM 3D GLB + object poses"))
